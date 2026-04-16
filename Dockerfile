FROM node:20-bookworm

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3001
ENV PYTHON_BIN=python3
ENV AI_PYTHON_BIN=python3

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ffmpeg \
  && rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json ./
COPY client/package.json ./client/package.json
COPY server/package.json ./server/package.json

RUN npm install

COPY . .

RUN pip3 install --no-cache-dir -r server/requirements.txt
RUN npm run build
RUN npm prune --omit=dev

EXPOSE 3001

CMD ["npm", "--workspace", "server", "run", "start"]
