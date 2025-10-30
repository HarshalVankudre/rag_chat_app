#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# --- CONFIGURATION ---
# Set these variables
export IMAGE_NAME="registry.digitalocean.com/ruekogpt1/rag-chat-app:latest"
export SERVER_USER="root"
export SERVER_IP="138.197.185.107" # <-- Put your server's IP here
export CONTAINER_NAME="rag-chat-container"
export APP_PORT="8501"
export HEALTH_CHECK_URL="http://127.0.0.1:${APP_PORT}/_stcore/health"
export HEALTH_CHECK_TIMEOUT=30  # seconds to wait for health check
export MAX_RETRIES=3
# ---------------------

# --- ANSI Colors ---
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[1;34m'
readonly CYAN='\033[1;36m'
readonly MAGENTA='\033[1;35m'
readonly RESET='\033[0m'
readonly BOLD='\033[1m'

# --- Logging Functions ---
print_step() {
    printf "\n${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
    printf "${CYAN}${BOLD}  %s${RESET}\n" "$1"
    printf "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
}

print_success() {
    printf "${GREEN}${BOLD}✓${RESET} ${GREEN}%s${RESET}\n" "$1"
}

print_error() {
    printf "${RED}${BOLD}✗${RESET} ${RED}%s${RESET}\n" "$1" >&2
}

print_warning() {
    printf "${YELLOW}${BOLD}⚠${RESET} ${YELLOW}%s${RESET}\n" "$1"
}

print_info() {
    printf "${BLUE}ℹ${RESET} ${BLUE}%s${RESET}\n" "$1"
}

print_timestamp() {
    printf "${MAGENTA}[%s]${RESET} " "$(date '+%Y-%m-%d %H:%M:%S')"
}

print_with_timestamp() {
    print_timestamp
    printf "%s\n" "$1"
}

# --- Error Handling ---
cleanup_on_error() {
    local exit_code=$?
    print_error "Deployment failed with exit code: $exit_code"
    print_info "Attempting cleanup..."
    
    # Try to show container logs if container exists
    ssh -o ConnectTimeout=5 "${SERVER_USER}@${SERVER_IP}" << 'ENDSSH' 2>/dev/null || true
        export CONTAINER_NAME="rag-chat-container"
        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "=== Container logs (last 50 lines) ==="
            docker logs --tail 50 "${CONTAINER_NAME}" 2>/dev/null || true
        fi
ENDSSH
    
    exit $exit_code
}

trap cleanup_on_error ERR
trap 'print_error "Script interrupted"; exit 130' INT TERM

# --- Validation Functions ---
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "Required command '$1' not found. Please install it first."
        exit 1
    fi
}

validate_config() {
    local errors=0
    
    if [[ -z "${IMAGE_NAME:-}" ]]; then
        print_error "IMAGE_NAME is not set"
        ((errors++))
    fi
    
    if [[ -z "${SERVER_USER:-}" ]]; then
        print_error "SERVER_USER is not set"
        ((errors++))
    fi
    
    if [[ -z "${SERVER_IP:-}" ]]; then
        print_error "SERVER_IP is not set"
        ((errors++))
    fi
    
    if [[ -z "${CONTAINER_NAME:-}" ]]; then
        print_error "CONTAINER_NAME is not set"
        ((errors++))
    fi
    
    if [[ $errors -gt 0 ]]; then
        print_error "Configuration validation failed. Please fix the errors above."
        exit 1
    fi
    
    print_success "Configuration validated"
}

check_docker_auth() {
    print_info "Checking Docker registry authentication..."
    
    # Check if we can access the registry (without actually pulling)
    if docker manifest inspect "${IMAGE_NAME}" &> /dev/null; then
        print_success "Registry authentication OK"
    else
        print_warning "Could not verify registry access. This might be normal if the image doesn't exist yet."
        print_info "Make sure you're logged in: docker login registry.digitalocean.com"
    fi
}

check_ssh_connection() {
    print_info "Testing SSH connection to ${SERVER_USER}@${SERVER_IP}..."
    
    if ssh -o ConnectTimeout=10 -o BatchMode=yes "${SERVER_USER}@${SERVER_IP}" "echo 'SSH connection successful'" &> /dev/null; then
        print_success "SSH connection verified"
    else
        print_error "Cannot connect to server via SSH. Please check:"
        print_error "  1. Server IP: ${SERVER_IP}"
        print_error "  2. SSH key authentication is set up"
        print_error "  3. Server is accessible from your network"
        exit 1
    fi
}

check_server_docker() {
    print_info "Checking Docker availability on server..."
    
    if ssh "${SERVER_USER}@${SERVER_IP}" "command -v docker &> /dev/null && docker --version" &> /dev/null; then
        local docker_version
        docker_version=$(ssh "${SERVER_USER}@${SERVER_IP}" "docker --version" 2>/dev/null)
        print_success "Docker found on server: ${docker_version}"
    else
        print_error "Docker is not installed or not accessible on the server"
        exit 1
    fi
}

# --- Main Deployment Functions ---
build_image() {
    print_step "STEP 1: BUILDING DOCKER IMAGE"
    print_info "Image: ${IMAGE_NAME}"
    print_info "Building from current directory..."
    
    local build_start
    build_start=$(date +%s)
    
    if docker build -t "${IMAGE_NAME}" . --progress=plain; then
        local build_end
        build_end=$(date +%s)
        local build_duration=$((build_end - build_start))
        print_success "Build completed in ${build_duration} seconds"
    else
        print_error "Docker build failed"
        exit 1
    fi
}

push_image() {
    print_step "STEP 2: PUSHING TO DIGITALOCEAN REGISTRY"
    print_info "Pushing ${IMAGE_NAME}..."
    
    local push_start
    push_start=$(date +%s)
    
    if docker push "${IMAGE_NAME}"; then
        local push_end
        push_end=$(date +%s)
        local push_duration=$((push_end - push_start))
        print_success "Push completed in ${push_duration} seconds"
    else
        print_error "Docker push failed. Check your registry credentials."
        exit 1
    fi
}

deploy_to_server() {
    print_step "STEP 3: DEPLOYING ON SERVER (${SERVER_USER}@${SERVER_IP})"
    
    ssh "${SERVER_USER}@${SERVER_IP}" << ENDSSH
        set -euo pipefail
        
        export CONTAINER_NAME="${CONTAINER_NAME}"
        export IMAGE_NAME="${IMAGE_NAME}"
        export APP_PORT="${APP_PORT}"
        export HEALTH_CHECK_URL="${HEALTH_CHECK_URL}"
        export HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT}"
        
        # Logging functions
        print_info() {
            printf "\033[1;34mℹ\033[0m \033[1;34m%s\033[0m\n" "\$1"
        }
        print_success() {
            printf "\033[1;32m✓\033[0m \033[1;32m%s\033[0m\n" "\$1"
        }
        print_error() {
            printf "\033[0;31m✗\033[0m \033[0;31m%s\033[0m\n" "\$1" >&2
        }
        print_warning() {
            printf "\033[1;33m⚠\033[0m \033[1;33m%s\033[0m\n" "\$1"
        }
        
        # Stop old container gracefully
        print_info "Stopping old container (if running)..."
        if docker ps --format '{{.Names}}' | grep -q "^\\\${CONTAINER_NAME}\$"; then
            docker stop "\${CONTAINER_NAME}" --time 30 || true
            print_success "Old container stopped"
        else
            print_info "No running container found"
        fi
        
        # Remove old container
        print_info "Removing old container (if exists)..."
        if docker ps -a --format '{{.Names}}' | grep -q "^\\\${CONTAINER_NAME}\$"; then
            docker rm "\${CONTAINER_NAME}" || true
            print_success "Old container removed"
        else
            print_info "No container to remove"
        fi
        
        # Pull new image
        print_info "Pulling new image: \${IMAGE_NAME}"
        if docker pull "\${IMAGE_NAME}"; then
            print_success "Image pulled successfully"
        else
            print_error "Failed to pull image"
            exit 1
        fi
        
        # Get image ID for verification
        local image_id
        image_id=\$(docker images "\${IMAGE_NAME}" --format "{{.ID}}" | head -n1)
        print_info "Image ID: \${image_id:0:12}"
        
        # Run new container
        print_info "Starting new container..."
        if docker run -d --name "\${CONTAINER_NAME}" \\
          -p 127.0.0.1:\${APP_PORT}:\${APP_PORT} \\
          -e MONGO_URI="mongodb+srv://harshalvankudre_db_user:QYJ7qsxDnQg6OS8x@chatbot-1.acaznw5.mongodb.net/?retryWrites=true&w=majority&appName=chatbot-1" \\
          --restart always \\
          "\${IMAGE_NAME}"; then
            print_success "Container started"
        else
            print_error "Failed to start container"
            exit 1
        fi
        
        # Verify container is running
        print_info "Verifying container status..."
        sleep 2  # Give container a moment to start
        if docker ps --filter "name=\${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "\${CONTAINER_NAME}"; then
            docker ps --filter "name=\${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            print_success "Container is running"
        else
            print_error "Container failed to start"
            print_error "Container logs:"
            docker logs "\${CONTAINER_NAME}" 2>&1 | tail -20
            exit 1
        fi
        
        # Health check
        print_info "Waiting for application to be ready (timeout: \${HEALTH_CHECK_TIMEOUT}s)..."
        local check_count=0
        local max_checks=\$((HEALTH_CHECK_TIMEOUT / 2))
        local healthy=false
        
        while [ \$check_count -lt \$max_checks ]; do
            if curl -sf "\${HEALTH_CHECK_URL}" > /dev/null 2>&1; then
                healthy=true
                break
            fi
            sleep 2
            ((check_count++))
            printf "."
        done
        echo ""
        
        if [ "\$healthy" = true ]; then
            print_success "Application health check passed"
        else
            print_warning "Health check timeout - application may still be starting"
            print_info "Container logs (last 20 lines):"
            docker logs "\${CONTAINER_NAME}" 2>&1 | tail -20
        fi
        
        # Cleanup old images
        print_info "Cleaning up unused Docker images..."
        docker image prune -f --filter "dangling=true" > /dev/null 2>&1 || true
        print_success "Cleanup complete"
        
        print_success "Server deployment completed successfully!"
ENDSSH

    if [ $? -eq 0 ]; then
        print_success "Server deployment finished"
    else
        print_error "Server deployment failed"
        exit 1
    fi
}

show_deployment_summary() {
    print_step "STEP 4: DEPLOYMENT SUMMARY"
    
    print_info "Container: ${CONTAINER_NAME}"
    print_info "Image: ${IMAGE_NAME}"
    print_info "Server: ${SERVER_USER}@${SERVER_IP}"
    print_info "Port: ${APP_PORT}"
    
    # Try to get container status
    print_info "Fetching final container status..."
    ssh "${SERVER_USER}@${SERVER_IP}" "docker ps --filter 'name=${CONTAINER_NAME}' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || true
    
    print_success "Deployment completed successfully!"
    print_info "You can view logs with: ssh ${SERVER_USER}@${SERVER_IP} 'docker logs -f ${CONTAINER_NAME}'"
}

# --- Main Execution ---
main() {
    print_step "🚀 DEPLOYMENT SCRIPT STARTED"
    print_with_timestamp "Starting deployment process..."
    
    # Pre-flight checks
    print_step "PRE-FLIGHT CHECKS"
    print_info "Checking required tools..."
    check_command "docker"
    check_command "ssh"
    check_command "curl"
    print_success "All required tools are available"
    
    # Validate configuration
    validate_config
    
    # Check Docker registry authentication
    check_docker_auth
    
    # Check SSH connection
    check_ssh_connection
    
    # Check Docker on server
    check_server_docker
    
    # Main deployment steps
    build_image
    push_image
    deploy_to_server
    show_deployment_summary
    
    print_step "✅ DEPLOYMENT COMPLETED SUCCESSFULLY"
    print_with_timestamp "All steps completed!"
}

# Run main function
main "$@"
