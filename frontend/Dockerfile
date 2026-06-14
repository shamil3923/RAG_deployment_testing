FROM node:20-bullseye-slim

ARG VITE_API_BASE
ENV VITE_API_BASE=${VITE_API_BASE}

WORKDIR /app

# Install dependencies
COPY frontend/package.json frontend/package-lock.json* ./
RUN corepack enable && npm ci --silent || npm install --silent

# Copy source
COPY frontend/ /app/

# Build (Vite will inline VITE_API_BASE into the built code)
RUN VITE_API_BASE=${VITE_API_BASE} npm run build

# Serve with a lightweight static server
RUN npm install -g serve --silent

EXPOSE 5173
CMD ["serve", "-s", "dist", "-l", "5173"]
