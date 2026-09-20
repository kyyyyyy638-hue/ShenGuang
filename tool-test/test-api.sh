#!/usr/bin/env bash
# 独立验证智谱 API 连通性 —— 不经过 LangChain，隔离变量
#
# 用法：先在 .env 里填好 OPENAI_API_KEY，然后
#   bash test-api.sh
#
# 为什么要有这个脚本：
#   如果这一步就失败，问题在「key / 模型名 / 账号额度」；
#   如果这一步成功但 node 脚本失败，问题在 LangChain 传参。
#   先切一刀，省得在两个层面之间来回猜。
set -uo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

END="${OPENAI_BASE_URL:-https://open.bigmodel.cn/api/paas/v4}"
M="${MODEL_NAME:-glm-5.3}"
EFFORT="${REASONING_EFFORT:-low}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[X] .env 里的 OPENAI_API_KEY 还是空的 —— 先填上智谱的 API Key 再跑"
  exit 1
fi

echo "[i] 端点  : $END"
echo "[i] 模型  : $M"
echo "[i] 思考档: $EFFORT"
echo "[i] 请求中（非流式，便于看完整返回）..."
echo ""

# 说明：这里去掉了原 curl 里的 "temperature": 1.0。
# GLM-5.3 是强制思考的推理模型，官方示例不传 temperature。
# 如果你想试流式，加上 "stream": true，然后去掉下面的 -sS 用普通 curl 看 SSE。
curl -sS -X POST "$END/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -d "{
    \"model\": \"$M\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"你是一个有用的AI助手。\"},
      {\"role\": \"user\", \"content\": \"你好，请介绍一下自己。\"}
    ],
    \"thinking\": {\"type\": \"enabled\"},
    \"reasoning_effort\": \"$EFFORT\"
  }"

echo ""
