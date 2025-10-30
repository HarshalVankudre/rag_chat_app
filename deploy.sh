#!/bin/bash

###############################################################################
# Deployment Script for RAG Chat App
# 
# This script builds a Docker image, pushes it to DigitalOcean registry,
# and deploys it to a remote server.
#
# Features:
# - Comprehensive error handling and logging
# - Pre-flight checks
# - Health checks after deployment
# - Rollback capability
# - Configuration validation
###############################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures
IFS=$'\n\t'        # Internal Field Separator for safer word splitting

###############################################################################
# CONFIGURATION
###############################################################################

# Docker image configuration
export REGISTRY_BASE="${REGISTRY_BASE:-registry.digitalocean.com/ruekogpt1/rag-chat-app}"
# Use git commit SHA for tagging, but always rebuild from scratch
export GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"
export IMAGE_TAG="${IMAGE_TAG:-${GIT_SHA}}"
export IMAGE_NAME="${IMAGE_NAME:-${REGISTRY_BASE}:${IMAGE_TAG}}"
export IMAGE_LATEST="${IMAGE_LATEST:-${REGISTRY_BASE}:latest}"
# Always perform a full rebuild (no caching)
export BUILD_PLATFORM="${BUILD_PLATFORM:-linux/amd64}"

# Server configuration
export SERVER_USER="${SERVER_USER:-root}"
export SERVER_IP="${SERVER_IP:-138.197.185.107}"
export SERVER_PORT="${SERVER_PORT:-22}"
export CONTAINER_NAME="${CONTAINER_NAME:-rag-chat-container}"

# Application configuration
export APP_PORT="${APP_PORT:-8501}"
# Default to 0.0.0.0 to bind to all interfaces (matches Streamlit default in Dockerfile)
export APP_HOST="${APP_HOST:-0.0.0.0}"
export MONGO_URI="${MONGO_URI:-mongodb+srv://harshalvankudre_db_user:QYJ7qsxDnQg6OS8x@chatbot-1.acaznw5.mongodb.net/?retryWrites=true&w=majority&appName=chatbot-1}"

# Deployment options
export HEALTH_CHECK_ENABLED="${HEALTH_CHECK_ENABLED:-true}"
# Health check URL: use SERVER_IP for external access (APP_HOST=0.0.0.0 allows this)
# If APP_HOST is 127.0.0.1, health check will fail unless run from server itself
export HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://${SERVER_IP}:${APP_PORT}}"
export HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-60}"
export ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"
export CLEANUP_OLD_IMAGES="${CLEANUP_OLD_IMAGES:-true}"

# Logging configuration
export LOG_DIR="${LOG_DIR:-./logs}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/deploy-$(date +%Y%m%d-%H%M%S).log}"
export VERBOSE="${VERBOSE:-false}"

###############################################################################
# GLOBAL VARIABLES
###############################################################################

DEPLOYMENT_START_TIME=$(date +%s)
PREVIOUS_IMAGE_TAG=""
DEPLOYMENT_FAILED=false
CLEANUP_FUNCTIONS=()

###############################################################################
# LOGGING FUNCTIONS
###############################################################################

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Log with timestamp
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_entry="[${timestamp}] [${level}] ${message}"
    
    echo "${log_entry}" | tee -a "${LOG_FILE}"
}

# Colored output functions
print_step() {
    local message="$1"
    echo ""
    echo -e "\033[1;36m═══════════════════════════════════════════════════════════\033[0m"
    echo -e "\033[1;36m  ${message}\033[0m"
    echo -e "\033[1;36m═══════════════════════════════════════════════════════════\033[0m"
    log "STEP" "${message}"
}

print_success() {
    local message="$1"
    echo -e "\033[1;32m✅ ${message}\033[0m"
    log "SUCCESS" "${message}"
}

print_error() {
    local message="$1"
    echo -e "\033[1;31m❌ ERROR: ${message}\033[0m" >&2
    log "ERROR" "${message}"
}

print_warning() {
    local message="$1"
    echo -e "\033[1;33m⚠️  WARNING: ${message}\033[0m"
    log "WARNING" "${message}"
}

print_info() {
    local message="$1"
    echo -e "\033[1;34mℹ️  ${message}\033[0m"
    log "INFO" "${message}"
}

print_debug() {
    if [[ "${VERBOSE}" == "true" ]]; then
        local message="$1"
        echo -e "\033[0;36m🔍 DEBUG: ${message}\033[0m"
        log "DEBUG" "${message}"
    fi
}

###############################################################################
# ERROR HANDLING AND CLEANUP
###############################################################################

# Cleanup function to be called on exit
cleanup() {
    local exit_code=$?
    
    if [[ ${exit_code} -ne 0 ]] || [[ "${DEPLOYMENT_FAILED}" == "true" ]]; then
        print_error "Deployment failed or was interrupted!"
        DEPLOYMENT_FAILED=true
        
        # Execute cleanup functions in reverse order
        local i=${#CLEANUP_FUNCTIONS[@]}
        while [[ $i -gt 0 ]]; do
            i=$((i-1))
            ${CLEANUP_FUNCTIONS[$i]} || true
        done
        
        if [[ "${ROLLBACK_ON_FAILURE}" == "true" ]] && [[ -n "${PREVIOUS_IMAGE_TAG}" ]]; then
            print_info "Attempting rollback to previous image..."
            rollback_deployment || true
        fi
    fi
    
    local duration=$(( $(date +%s) - DEPLOYMENT_START_TIME ))
    print_info "Deployment script completed in ${duration} seconds"
    print_info "Full log available at: ${LOG_FILE}"
    
    exit ${exit_code}
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Function to add cleanup tasks
add_cleanup() {
    CLEANUP_FUNCTIONS+=("$1")
}

###############################################################################
# VALIDATION FUNCTIONS
###############################################################################

validate_configuration() {
    print_step "Validating Configuration"
    
    local errors=0
    
    # Validate required variables
    if [[ -z "${IMAGE_NAME}" ]]; then
        print_error "IMAGE_NAME is not set"
        errors=$((errors + 1))
    fi
    
    if [[ -z "${SERVER_USER}" ]]; then
        print_error "SERVER_USER is not set"
        errors=$((errors + 1))
    fi
    
    if [[ -z "${SERVER_IP}" ]]; then
        print_error "SERVER_IP is not set"
        errors=$((errors + 1))
    fi
    
    if [[ -z "${CONTAINER_NAME}" ]]; then
        print_error "CONTAINER_NAME is not set"
        errors=$((errors + 1))
    fi
    
    # Validate IP format (basic check)
    if ! [[ "${SERVER_IP}" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        print_warning "SERVER_IP format may be invalid: ${SERVER_IP}"
    fi
    
    # Validate port numbers
    if ! [[ "${SERVER_PORT}" =~ ^[0-9]+$ ]] || [[ "${SERVER_PORT}" -lt 1 ]] || [[ "${SERVER_PORT}" -gt 65535 ]]; then
        print_error "Invalid SERVER_PORT: ${SERVER_PORT}"
        errors=$((errors + 1))
    fi
    
    if ! [[ "${APP_PORT}" =~ ^[0-9]+$ ]] || [[ "${APP_PORT}" -lt 1 ]] || [[ "${APP_PORT}" -gt 65535 ]]; then
        print_error "Invalid APP_PORT: ${APP_PORT}"
        errors=$((errors + 1))
    fi
    
    # Validate APP_HOST and HEALTH_CHECK_URL consistency
    if [[ "${APP_HOST}" == "127.0.0.1" ]] && [[ "${HEALTH_CHECK_URL}" =~ ^http://${SERVER_IP} ]]; then
        print_warning "APP_HOST is set to 127.0.0.1 (localhost) but HEALTH_CHECK_URL uses SERVER_IP"
        print_warning "This will cause health checks to fail unless run from the server itself"
        print_info "Consider setting APP_HOST=0.0.0.0 to bind to all interfaces"
        print_info "Or set HEALTH_CHECK_URL=http://127.0.0.1:${APP_PORT} if health check runs on server"
    fi
    
    if [[ "${APP_HOST}" == "0.0.0.0" ]]; then
        print_debug "APP_HOST is 0.0.0.0 - container will be accessible from all interfaces"
    fi
    
    if [[ ${errors} -gt 0 ]]; then
        print_error "Configuration validation failed with ${errors} error(s)"
        exit 1
    fi
    
    print_success "Configuration validated successfully"
    print_debug "Image: ${IMAGE_NAME}"
    print_debug "Server: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
    print_debug "Container: ${CONTAINER_NAME}"
    print_debug "App binding: ${APP_HOST}:${APP_PORT}"
    print_debug "Health check: ${HEALTH_CHECK_URL}"
}

check_local_docker() {
    print_step "Checking Local Docker Environment"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    print_success "Docker is available and running"
    
    # Check Docker registry login
    if ! docker info | grep -q "Username"; then
        print_warning "Docker registry login may be required"
        print_info "Run: docker login ${REGISTRY_BASE%%/*}"
    fi
}

check_ssh_connection() {
    print_step "Checking SSH Connection"
    
    local test_command="echo 'SSH connection test'"
    
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes -p "${SERVER_PORT}" \
         "${SERVER_USER}@${SERVER_IP}" "${test_command}" &> /dev/null; then
        print_error "Cannot connect to server via SSH"
        print_info "Please ensure:"
        print_info "  1. SSH key is added to authorized_keys"
        print_info "  2. Server is accessible at ${SERVER_IP}:${SERVER_PORT}"
        print_info "  3. SSH agent is running (if using key forwarding)"
        exit 1
    fi
    
    print_success "SSH connection established"
}

check_server_docker() {
    print_step "Checking Server Docker Environment"
    
    ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" bash << 'ENDSSH'
        if ! command -v docker &> /dev/null; then
            echo "ERROR: Docker is not installed on server"
            exit 1
        fi
        
        if ! docker info &> /dev/null; then
            echo "ERROR: Docker daemon is not running on server"
            exit 1
        fi
        
        echo "SUCCESS: Docker is available on server"
ENDSSH
    
    if [[ $? -ne 0 ]]; then
        print_error "Server Docker check failed"
        exit 1
    fi
    
    print_success "Server Docker environment is ready"
}

###############################################################################
# DEPLOYMENT FUNCTIONS
###############################################################################

get_previous_image_tag() {
    print_debug "Checking for previous container image"
    
    PREVIOUS_IMAGE_TAG=$(ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" bash << ENDSSH
        docker inspect --format='{{.Config.Image}}' ${CONTAINER_NAME} 2>/dev/null || echo ""
ENDSSH
    )
    
    if [[ -n "${PREVIOUS_IMAGE_TAG}" ]]; then
        print_info "Found previous image: ${PREVIOUS_IMAGE_TAG}"
    else
        print_info "No previous container found (first deployment)"
    fi
}

optimize_docker_for_push() {
    print_debug "Optimizing Docker settings for faster push..."
    
    # Don't force BuildKit - let Docker decide or use it conditionally
    # BuildKit can cause issues with Windows paths and cloud builders
    # Push will still be fast as Docker pushes layers in parallel by default
    
    # Check if buildx is available but avoid cloud builders
    if command -v docker &> /dev/null && docker buildx version &> /dev/null 2>&1; then
        print_debug "Docker Buildx available"
        # Check if default builder uses local driver (not cloud)
        local builders=$(docker buildx ls 2>/dev/null || echo "")
        if echo "${builders}" | grep -q "default.*docker"; then
            print_debug "Local builder found"
        elif echo "${builders}" | grep -q "cloud"; then
            print_debug "Cloud builder detected - will use regular docker build"
        fi
    fi
}

build_docker_image() {
    print_step "Building Docker Image"
    print_info "Image: ${IMAGE_NAME} (git: ${GIT_SHA})"
    print_info "Platform: ${BUILD_PLATFORM}"
    
    # Clean up Python cache files that can cause BuildKit issues
    print_debug "Cleaning up Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Optimize Docker for build and push
    optimize_docker_for_push
    
    # Remove any local images for this tag to guarantee a clean build
    print_info "Removing local images for a clean rebuild..."
    docker image rm -f "${IMAGE_NAME}" "${IMAGE_LATEST}" >/dev/null 2>&1 || true
    docker builder prune -af >/dev/null 2>&1 || true
    docker image prune -af >/dev/null 2>&1 || true
    
    local build_start=$(date +%s)
    
    # Default to using Buildx when available and local
    local use_buildx=false
    if command -v docker &> /dev/null && docker buildx version &> /dev/null 2>&1; then
        local builder_info=$(docker buildx inspect --bootstrap 2>/dev/null || echo "")
        if [[ -n "${builder_info}" ]] && ! echo "${builder_info}" | grep -qi "cloud"; then
            use_buildx=true
        else
            print_warning "Buildx available but no local builder detected - falling back to regular docker build"
        fi
    fi
    
    # Allow users to opt out of buildx if needed
    if [[ "${FORCE_LEGACY_BUILD:-false}" == "true" ]]; then
        use_buildx=false
    fi
    
    # Always enable BuildKit for faster builds unless explicitly disabled
    export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
    
    # Try BuildKit/Buildx first (much faster builds with better caching)
    # Fallback to legacy build if BuildKit fails
    if [[ "${use_buildx}" == "true" ]]; then
        print_info "Using Docker Buildx (full rebuild, no cache)"
        
        # Build with Buildx optimizations (compression flag requires Docker 24.0+)
        # Try with compression first, fallback without it if not supported
        if ! docker buildx build \
            --platform="${BUILD_PLATFORM}" \
            --no-cache \
            --load \
            -t "${IMAGE_NAME}" \
            -t "${IMAGE_LATEST}" \
            . 2>&1 | tee -a "${LOG_FILE}"; then
            print_warning "Buildx build failed, falling back to regular docker build"
            DOCKER_BUILDKIT=0 docker build --no-cache -t "${IMAGE_NAME}" -t "${IMAGE_LATEST}" . 2>&1 | tee -a "${LOG_FILE}" || {
                print_error "Docker build failed"
                exit 1
            }
        fi
    else
        # Try BuildKit first (faster builds, better caching)
        print_info "Using Docker BuildKit"
        if ! DOCKER_BUILDKIT=1 docker build --no-cache -t "${IMAGE_NAME}" -t "${IMAGE_LATEST}" . 2>&1 | tee -a "${LOG_FILE}"; then
            print_warning "BuildKit build failed, falling back to legacy build engine..."
            if ! DOCKER_BUILDKIT=0 docker build --no-cache -t "${IMAGE_NAME}" -t "${IMAGE_LATEST}" . 2>&1 | tee -a "${LOG_FILE}"; then
                print_error "Docker build failed with both BuildKit and legacy engine"
                exit 1
            fi
        fi
    fi
    
    local build_duration=$(( $(date +%s) - build_start ))
    print_success "Build completed in ${build_duration} seconds"
    
    # Tag with git SHA for content-addressed caching
    if [[ "${GIT_SHA}" != "latest" ]]; then
        print_info "Tagged as ${IMAGE_TAG} and latest"
    fi
    
    # Add cleanup to remove local image if push fails
    add_cleanup "docker rmi ${IMAGE_NAME} ${IMAGE_LATEST} || true"
}

# Helper function to format bytes to human-readable format
format_size() {
    local bytes=$1
    if command -v numfmt &> /dev/null; then
        numfmt --to=iec-i --suffix=B "${bytes}"
    else
        # Fallback: approximate conversion
        if [[ ${bytes} -lt 1048576 ]]; then
            echo "$((bytes / 1024))KB"
        elif [[ ${bytes} -lt 1073741824 ]]; then
            echo "$((bytes / 1024 / 1024))MB"
        else
            echo "$((bytes / 1024 / 1024 / 1024))GB"
        fi
    fi
}

push_docker_image() {
    print_step "Pushing Image to Registry"
    print_info "Registry: ${REGISTRY_BASE%%/*}"
    print_info "Tags: ${IMAGE_TAG} (git SHA) and latest"
    
    # Remove any existing manifests in the registry to ensure a clean push
    print_info "Removing existing registry manifests (if any)..."
    docker buildx imagetools rm "${IMAGE_NAME}" >/dev/null 2>&1 || docker manifest rm "${IMAGE_NAME}" >/dev/null 2>&1 || true
    docker buildx imagetools rm "${IMAGE_LATEST}" >/dev/null 2>&1 || docker manifest rm "${IMAGE_LATEST}" >/dev/null 2>&1 || true
    
    # Optimize Docker for push
    optimize_docker_for_push
    
    # Get image size and layer information before push
    local image_size=$(docker image inspect "${IMAGE_NAME}" --format='{{.Size}}' 2>/dev/null || echo "0")
    if [[ "${image_size}" != "0" ]]; then
        print_info "Image size: $(format_size "${image_size}")"
        
        # Count layers
        local layer_count=$(docker image inspect "${IMAGE_NAME}" --format='{{len .RootFS.Layers}}' 2>/dev/null || echo "0")
        if [[ "${layer_count}" != "0" ]]; then
            print_info "Number of layers: ${layer_count}"
        fi
        
        # Estimate push time (rough estimate: ~10-50 MB/s depending on connection)
        local estimated_speed_mbps=20  # Conservative estimate
        local estimated_time=$((image_size / 1024 / 1024 / estimated_speed_mbps))
        if [[ ${estimated_time} -gt 0 ]]; then
            print_info "Estimated push time: ~${estimated_time} seconds (at ${estimated_speed_mbps}MB/s)"
        fi
    fi
    
    print_info "Uploading layers (this may take a while)..."
    print_info "Tip: Docker pushes layers in parallel for faster upload"
    local push_start=$(date +%s)
    
    # Track current layer being pushed
    local current_layer=""
    local layer_num=0
    
    # Docker push shows progress automatically
    # Docker already pushes layers in parallel - no special command needed
    # Parse output to show per-layer progress with sizes
    if ! docker push "${IMAGE_NAME}" 2>&1 | tee -a "${LOG_FILE}" | while IFS= read -r line || [[ -n "${line}" ]]; do
        # Log everything to file
        echo "${line}" >> "${LOG_FILE}"
        
        # Parse layer hash (sha256:...)
        if [[ "${line}" =~ (sha256:[a-f0-9]+) ]]; then
            local layer_hash="${BASH_REMATCH[1]}"
            if [[ "${layer_hash}" != "${current_layer}" ]]; then
                current_layer="${layer_hash}"
                layer_num=$((layer_num + 1))
                echo ""
                echo -e "\033[1;34m  Layer ${layer_num}: ${layer_hash:0:19}...\033[0m"
            fi
        fi
        
        # Parse progress line with size information
        # Format: "sha256:abc123: Pushing   [========>] 45.2MB/120.5MB" or "sha256:abc123: Pushing   45.2MB/120.5MB"
        if [[ "${line}" =~ (Pushing|Pulling|Pulled|Pull|Waiting|Preparing|Verifying|Mounted|Already)[[:space:]]+.*([0-9]+\.[0-9]+)([KMGT]?B)/([0-9]+\.[0-9]+)([KMGT]?B) ]]; then
            local status="${BASH_REMATCH[1]}"
            local current="${BASH_REMATCH[2]}${BASH_REMATCH[3]}"
            local total="${BASH_REMATCH[4]}${BASH_REMATCH[5]}"
            
            # Extract progress bar if present
            local progress_bar=""
            if [[ "${line}" =~ \[([^\]]+)\] ]]; then
                progress_bar="${BASH_REMATCH[1]}"
                echo -e "    \033[1;36m${status}\033[0m ${current}/${total} [${progress_bar}]"
            else
                echo -e "    \033[1;36m${status}\033[0m ${current}/${total}"
            fi
        elif [[ "${line}" =~ (Pushing|Pulling|Pulled|Pull|Waiting|Preparing|Verifying|Mounted|Already)[[:space:]]+([0-9]+\.[0-9]+)([KMGT]?B) ]]; then
            # Single size (total size shown)
            local status="${BASH_REMATCH[1]}"
            local size="${BASH_REMATCH[2]}${BASH_REMATCH[3]}"
            echo -e "    \033[1;36m${status}\033[0m ${size}"
        elif [[ "${line}" =~ (digest:|pushed|Pushed|latest|tag:) ]]; then
            # Success indicators
            echo -e "    \033[1;32m✓ ${line}\033[0m"
        elif [[ "${line}" =~ (already exists|Already exists) ]]; then
            # Layer already exists
            echo -e "    \033[1;33m✓ ${line}\033[0m"
        elif [[ "${line}" =~ (error|Error|ERROR|failed|Failed|denied|unauthorized) ]]; then
            # Always show errors
            echo -e "    \033[1;31m✗ ${line}\033[0m" >&2
        elif [[ "${VERBOSE}" == "true" ]]; then
            # Show all output in verbose mode
            echo "    ${line}"
        fi
    done; then
        print_error "Docker push failed"
        print_info "Check your registry credentials and network connection"
        print_info "Check the log file for details: ${LOG_FILE}"
        exit 1
    fi
    
    local push_duration=$(( $(date +%s) - push_start ))
    echo ""
    print_success "Push completed in ${push_duration} seconds"
    
    # Also push latest tag (will reuse layers, very fast)
    if [[ "${IMAGE_TAG}" != "latest" ]]; then
        print_info "Updating latest tag..."
        local latest_push_start=$(date +%s)
        if docker push "${IMAGE_LATEST}" 2>&1 | tee -a "${LOG_FILE}"; then
            local latest_push_duration=$(( $(date +%s) - latest_push_start ))
            print_success "Latest tag updated in ${latest_push_duration} seconds"
        else
            print_warning "Failed to push latest tag (non-critical)"
        fi
    fi
    
    # Show final image information
    print_info "Image pushed successfully:"
    print_info "  - ${IMAGE_NAME} (git SHA)"
    print_info "  - ${IMAGE_LATEST} (latest)"
}

deploy_to_server() {
    print_step "Deploying to Server"
    print_info "Target: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
    
    # Get previous image before deploying
    get_previous_image_tag
    
    ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" bash << ENDSSH
        set -euo pipefail
        IFS=$'\n\t'
        
        # Configuration
        export CONTAINER_NAME="${CONTAINER_NAME}"
        export IMAGE_NAME="${IMAGE_NAME}"
        export REGISTRY_BASE="${REGISTRY_BASE}"
        export APP_PORT="${APP_PORT}"
        export APP_HOST="${APP_HOST}"
        export MONGO_URI="${MONGO_URI}"
        
        # Logging functions
        log() {
            local level="\$1"
            shift
            local message="\$*"
            local timestamp=\$(date '+%Y-%m-%d %H:%M:%S')
            echo "[\${timestamp}] [\${level}] \${message}"
        }
        
        print_info() {
            echo -e "\033[1;34mℹ️  \$1\033[0m"
            log "INFO" "\$1"
        }
        
        print_success() {
            echo -e "\033[1;32m✅ \$1\033[0m"
            log "SUCCESS" "\$1"
        }
        
        print_error() {
            echo -e "\033[1;31m❌ ERROR: \$1\033[0m" >&2
            log "ERROR" "\$1"
            exit 1
        }
        
        # Stop and remove old container
        print_info "Stopping old container (if exists)..."
        if docker ps -a --format '{{.Names}}' | grep -q "^\${CONTAINER_NAME}\$"; then
            docker stop "\${CONTAINER_NAME}" || true
            docker rm "\${CONTAINER_NAME}" || true
            print_success "Old container removed"
        else
            print_info "No existing container found"
        fi
        
        # Remove any existing images for this repository to enforce a fresh pull
        print_info "Removing existing server images for \${REGISTRY_BASE}..."
        existing_images=$(docker images "\${REGISTRY_BASE}" --format '{{.Repository}}:{{.Tag}}')
        if [[ -n "\${existing_images}" ]]; then
            for img in \${existing_images}; do
                docker rmi -f "\${img}" || true
            done
        fi
        docker image prune -af >/dev/null 2>&1 || true
        
        # Pull new image with progress display
        print_info "Pulling new image: \${IMAGE_NAME}"
        
        # Track current layer being pulled
        current_layer=""
        layer_num=0
        
        if ! docker pull "\${IMAGE_NAME}" 2>&1 | while IFS= read -r line || [[ -n "\${line}" ]]; do
            # Parse layer hash (sha256:...)
            if [[ "\${line}" =~ (sha256:[a-f0-9]+) ]]; then
                layer_hash="\${BASH_REMATCH[1]}"
                if [[ "\${layer_hash}" != "\${current_layer}" ]]; then
                    current_layer="\${layer_hash}"
                    layer_num=\$((layer_num + 1))
                    echo ""
                    echo -e "\033[1;34m  Layer \${layer_num}: \${layer_hash:0:19}...\033[0m"
                fi
            fi
            
            # Parse progress line with size information
            # Format: "sha256:abc123: Pulling   [========>] 45.2MB/120.5MB" or "sha256:abc123: Pulling   45.2MB/120.5MB"
            if [[ "\${line}" =~ (Pushing|Pulling|Pulled|Pull|Waiting|Preparing|Verifying|Mounted|Already)[[:space:]]+.*([0-9]+\.[0-9]+)([KMGT]?B)/([0-9]+\.[0-9]+)([KMGT]?B) ]]; then
                status="\${BASH_REMATCH[1]}"
                current="\${BASH_REMATCH[2]}\${BASH_REMATCH[3]}"
                total="\${BASH_REMATCH[4]}\${BASH_REMATCH[5]}"
                
                # Extract progress bar if present
                progress_bar=""
                if [[ "\${line}" =~ \[([^\]]+)\] ]]; then
                    progress_bar="\${BASH_REMATCH[1]}"
                    echo -e "    \033[1;36m\${status}\033[0m \${current}/\${total} [\${progress_bar}]"
                else
                    echo -e "    \033[1;36m\${status}\033[0m \${current}/\${total}"
                fi
            elif [[ "\${line}" =~ (Pulling|Pulled|Pull|Waiting|Preparing|Verifying|Mounted|Already)[[:space:]]+([0-9]+\.[0-9]+)([KMGT]?B) ]]; then
                # Single size (total size shown)
                status="\${BASH_REMATCH[1]}"
                size="\${BASH_REMATCH[2]}\${BASH_REMATCH[3]}"
                echo -e "    \033[1;36m\${status}\033[0m \${size}"
            elif [[ "\${line}" =~ (digest:|pulled|Pulled|latest|tag:) ]]; then
                # Success indicators
                echo -e "    \033[1;32m✓ \${line}\033[0m"
            elif [[ "\${line}" =~ (already exists|Already exists) ]]; then
                # Layer already exists
                echo -e "    \033[1;33m✓ \${line}\033[0m"
            elif [[ "\${line}" =~ (error|Error|ERROR|failed|Failed|denied|unauthorized) ]]; then
                # Always show errors
                echo -e "    \033[1;31m✗ \${line}\033[0m" >&2
            fi
        done; then
            print_error "Failed to pull image: \${IMAGE_NAME}"
        fi
        echo ""
        print_success "Image pulled successfully"
        
        # Run new container
        print_info "Starting new container..."
        if ! docker run -d \
            --name "\${CONTAINER_NAME}" \
            -p "\${APP_HOST}:\${APP_PORT}:\${APP_PORT}" \
            -e MONGO_URI="\${MONGO_URI}" \
            --restart always \
            "\${IMAGE_NAME}"; then
            print_error "Failed to start container"
        fi
        
        print_success "Container started successfully"
        
        # Wait for container to be running
        print_info "Waiting for container to be ready..."
        sleep 3
        
        # Verify container status
    print_info "Verifying container status..."
        if ! docker ps --filter "name=\${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "\${CONTAINER_NAME}"; then
            print_error "Container is not running"
        fi
        
        # Show container status
        docker ps --filter "name=\${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        
        # Show container logs (last 20 lines)
        print_info "Recent container logs:"
        docker logs --tail 20 "\${CONTAINER_NAME}" || true
        
        # Cleanup old images
        if [[ "${CLEANUP_OLD_IMAGES}" == "true" ]]; then
            print_info "Cleaning up unused Docker images..."
            docker image prune -af --filter "until=24h" || true
            print_success "Cleanup completed"
        fi
        
        print_success "Server deployment completed successfully"
ENDSSH
    
    if [[ $? -ne 0 ]]; then
        print_error "Server deployment failed"
        DEPLOYMENT_FAILED=true
        exit 1
    fi
    
    print_success "Deployment to server completed"
}

health_check() {
    if [[ "${HEALTH_CHECK_ENABLED}" != "true" ]]; then
        print_info "Health check disabled, skipping..."
        return 0
    fi
    
    print_step "Performing Health Check"
    print_info "Checking: ${HEALTH_CHECK_URL}"
    
    local check_start=$(date +%s)
    local check_timeout=${HEALTH_CHECK_TIMEOUT}
    local elapsed=0
    local max_attempts=10
    local attempt=0
    
    while [[ ${elapsed} -lt ${check_timeout} ]] && [[ ${attempt} -lt ${max_attempts} ]]; do
        attempt=$((attempt + 1))
        print_info "Health check attempt ${attempt}/${max_attempts}..."
        
        # Try to check if the service is responding
        # Use curl if available, otherwise try to connect via SSH and check container
        if command -v curl &> /dev/null; then
            if curl -f -s --max-time 5 "${HEALTH_CHECK_URL}" > /dev/null 2>&1; then
                print_success "Health check passed (attempt ${attempt})"
                return 0
            fi
        else
            # Fallback: check container status via SSH
            if ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" \
                   "docker exec ${CONTAINER_NAME} echo 'Container is responsive'" &> /dev/null; then
                print_success "Container health check passed (attempt ${attempt})"
                return 0
            fi
        fi
        
        sleep 3
        elapsed=$(( $(date +%s) - check_start ))
    done
    
    print_warning "Health check did not pass within ${check_timeout} seconds"
    print_info "This may be normal if the application takes time to start"
    print_info "Please verify manually: ${HEALTH_CHECK_URL}"
    return 1
}

rollback_deployment() {
    if [[ -z "${PREVIOUS_IMAGE_TAG}" ]]; then
        print_info "No previous image to rollback to"
        return 0
    fi
    
    print_step "Rolling Back to Previous Image"
    print_info "Previous image: ${PREVIOUS_IMAGE_TAG}"
    
    ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" bash << ENDSSH
        set -euo pipefail
        
        export CONTAINER_NAME="${CONTAINER_NAME}"
        export PREVIOUS_IMAGE="${PREVIOUS_IMAGE_TAG}"
        export APP_PORT="${APP_PORT}"
        export APP_HOST="${APP_HOST}"
        export MONGO_URI="${MONGO_URI}"
        
        echo "Stopping current container..."
        docker stop "\${CONTAINER_NAME}" || true
        docker rm "\${CONTAINER_NAME}" || true
        
        echo "Starting previous container..."
        docker run -d \
            --name "\${CONTAINER_NAME}" \
            -p "\${APP_HOST}:\${APP_PORT}:\${APP_PORT}" \
            -e MONGO_URI="\${MONGO_URI}" \
            --restart always \
            "\${PREVIOUS_IMAGE}"
        
        echo "Rollback completed"
ENDSSH

    if [[ $? -eq 0 ]]; then
        print_success "Rollback completed successfully"
    else
        print_error "Rollback failed"
        return 1
    fi
}

###############################################################################
# MAIN DEPLOYMENT FLOW
###############################################################################

main() {
    print_step "Starting Deployment Process"
    print_info "Log file: ${LOG_FILE}"
    print_info "Configuration:"
    print_info "  Image: ${IMAGE_NAME}"
    print_info "  Server: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
    print_info "  Container: ${CONTAINER_NAME}"
    print_info "  App Port: ${APP_PORT}"
    
    # Pre-flight checks
    validate_configuration
    check_local_docker
    check_ssh_connection
    check_server_docker
    
    # Deployment steps
    build_docker_image
    push_docker_image
    deploy_to_server
    
    # Post-deployment checks
    health_check || print_warning "Health check failed, but deployment completed"
    
    # Success
    print_step "Deployment Complete!"
    local duration=$(( $(date +%s) - DEPLOYMENT_START_TIME ))
    print_success "Total deployment time: ${duration} seconds"
    print_info "Application URL: ${HEALTH_CHECK_URL}"
    print_info "View logs: docker logs -f ${CONTAINER_NAME}"
    print_info "Full deployment log: ${LOG_FILE}"
    
    DEPLOYMENT_FAILED=false
}

# Run main function
main "$@"
