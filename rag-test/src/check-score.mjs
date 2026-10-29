// 一次性校验：MemoryVectorStore 返回的 score 是「余弦相似度」还是「距离」？
// 文章里写的是 const similarity = 1 - score，如果 score 本身就是相似度，这句就反了。
import { embeddings } from "./models.mjs";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";

const text = "东东是光光最好的朋友，他是一个安静而聪明的男孩。东东喜欢读书和画画，他的画总是充满了想象力。虽然性格不同，但东东和光光从幼儿园就认识了，他们一起度过了无数个快乐的时光。";
const query = "东东和光光是怎么成为朋友的？";

const [vd] = await embeddings.embedDocuments([text]);
const vq = await embeddings.embedQuery(query);
const dot = vd.reduce((s, x, i) => s + x * vq[i], 0);
const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
const manual = dot / (norm(vd) * norm(vq));

const store = await MemoryVectorStore.fromDocuments([new Document({ pageContent: text })], embeddings);
const [, score] = (await store.similaritySearchWithScore(query, 1))[0];

console.log("向量维度            =", vd.length);
console.log("手工余弦相似度      =", manual.toFixed(6));
console.log("store 返回的 score  =", score.toFixed(6));
console.log("文章写法 1 - score  =", (1 - score).toFixed(6));
console.log("判定：", Math.abs(score - manual) < 1e-6
  ? "score 就是余弦相似度（越大越相似）→ 文章那句 1 - score 是反的"
  : "score 不是余弦相似度，需另作判断");
