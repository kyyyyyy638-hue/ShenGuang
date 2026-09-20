import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  // GLM-5.3 强制开启思考模式，关不掉。官方文档：不显式传 thinking.type=enabled
  // 请求会直接失败；reasoning_effort 默认 max，简单任务用 low 省 token 又快。
  // modelKwargs 是 LangChain 透传额外参数给 API 的通道。
  modelKwargs: {
    thinking: { type: 'enabled' },
    reasoning_effort: process.env.REASONING_EFFORT || 'low',
  },
});

const response = await model.invoke('介绍下自己');
console.log(response.content);
