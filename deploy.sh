#!/bin/bash
set -e # This makes the script exit if any command fails

# --- CONFIGURATION ---
# Set these variables
export IMAGE_NAME="registry.digitalocean.com/ruekogpt1/rag-chat-app:latest"
export SERVER_USER="root"
export SERVER_IP="138.197.185.107" # <-- Put your server's IP here
export CONTAINER_NAME="rag-chat-container"
# ---------------------

# --- Logging Helpers ---
# \033[1;36m = Bold Cyan
# \033[1;32m = Bold Green
# \033[1;34m = Bold Blue
# \033[0m    = Reset color
print_step() {
    printf "\n\033[1;36m--- %s ---\033[0m\n" "$1"
}
print_success() {
    printf "\033[1;32m✅ %s\033[0m\n" "$1"
}
print_info() {
    printf "\033[1;34mℹ️ %s\033[0m\n" "$1"
}
# ---------------------

print_step "STEP 1: BUILDING DOCKER IMAGE"
print_info "Image: $IMAGE_NAME"
print_info "This may take a moment..."
docker build -t $IMAGE_NAME .
print_success "Build complete."

print_step "STEP 2: PUSHING TO DIGITALOCEAN REGISTRY"
docker push $IMAGE_NAME
print_success "Push complete."

print_step "STEP 3: DEPLOYING ON SERVER ($SERVER_USER@$SERVER_IP)"

# This 'ssh' command logs into your server and runs all the commands
# inside the 'ENDSSH' block.
ssh $SERVER_USER@$SERVER_IP << 'ENDSSH'
    set -e # Exit on any error on the server

    # We must re-export these variables inside the SSH session
    export CONTAINER_NAME="rag-chat-container"
    export IMAGE_NAME="registry.digitalocean.com/ruekogpt1/rag-chat-app:latest"

    # Define the same logging functions on the server
    print_info() {
        printf "\033[1;34mℹ️ %s\033[0m\n" "$1"
    }
    print_success() {
        printf "\033[1;32m✅ %s\033[0m\n" "$1"
    }

    print_info "Logged in. Stopping old container (if it exists)..."
    # '|| true' means "don't fail if the container isn't running"
    docker stop $CONTAINER_NAME || true

    print_info "Removing old container (if it exists)..."
    docker rm $CONTAINER_NAME || true

    print_info "Pulling new image..."
    docker pull $IMAGE_NAME

    print_info "Running new container..."
    # This is your exact 'docker run' command
    docker run -d --name $CONTAINER_NAME \
      -p 127.0.0.1:8501:8501 \
      -e MONGO_URI="mongodb+srv://harshalvankudre_db_user:QYJ7qsxDnQg6OS8x@chatbot-1.acaznw5.mongodb.net/?retryWrites=true&w=majority&appName=chatbot-1" \
      --restart always \
      $IMAGE_NAME

    print_success "New container started."

    print_info "Verifying container status..."
    # This command will show us the running container
    docker ps --filter "name=$CONTAINER_NAME"

    print_info "Cleaning up old images on server..."
    # This removes any old, unused images to save disk space
    docker system prune -f

    print_success "Server deployment complete!"
ENDSSH

print_step "STEP 4: ALL DONE!"
print_success "FULL DEPLOYMENT FINISHED!"

