FROM node:20.18-alpine AS base

WORKDIR /app

RUN adduser -D appuser && chown appuser:appuser /app

USER appuser

COPY --chown=appuser:appuser package.json package-lock.json* .npmrc* ./
RUN npm install --legacy-peer-deps

COPY --chown=appuser:appuser . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
