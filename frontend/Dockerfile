FROM node:20.18-alpine AS base

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

RUN adduser -D appuser
USER appuser

COPY --chown=appuser:appuser . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
