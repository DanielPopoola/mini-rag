#!/bin/bash

# Deploy Mini RAG with Docker Compose

echo "🐳 Starting Mini RAG deployment..."

# Check if .env exists, if not copy from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration!"
fi

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
docker-compose exec -T backend curl -f http://localhost:9000/health || echo "Backend not ready yet"

echo "✅ Deployment complete!"
echo ""
echo "📊 Services:"
echo "  • Frontend: http://localhost:8501"
echo "  • Backend API: http://localhost:9000"
echo "  • API Docs: http://localhost:9000/docs"
echo "  • Qdrant: http://localhost:6333"
echo ""
echo "📝 View logs:"
echo "  docker-compose logs -f [service-name]"
echo ""
echo "🛑 To stop:"
echo "  docker-compose down"