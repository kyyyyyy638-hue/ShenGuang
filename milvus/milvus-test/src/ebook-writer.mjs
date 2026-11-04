// ============================================================
// 【这个文件干什么】
// 把一整本电子书（.epub 或 .txt）切碎、向量化，灌进 Milvus，
// 供后面做语义检索（比如问"段誉会什么武功"）。
//
// 为什么需要这一步：
//   问"这本书里 XX 是怎么回事"，用 MySQL 的 LIKE 根本没法查 ——
//   关键词定不出来。搜人名会命中所有出现处，搜"武功"全书都在讲。
//   向量检索是拿"问题的语义"比对"每段文字的语义"，不需要事先想关键词。
//
// 【运行方式】在 milvus-test 目录下，路径写在命令后面：
//     cd E:\代码项目\ShenGuang\milvus\milvus-test
//     node src/ebook-writer.mjs "D:\books\新婚_东风瘦.epub"
//
//   也可以走环境变量或默认路径：
//     $env:EBOOK_PATH="D:\books\x.epub"; node src/ebook-writer.mjs
//     node src/ebook-writer.mjs                 # 读 data/ebook.txt
//
// 【支持两种格式】
//   .epub —— 本质是 zip 包，内部是 xhtml。脚本会自动解压、
//            按 OPF 里 spine 的顺序逐章解析，保留章节号。
//   其他   —— 当纯文本读（UTF-8），整本视作一章。
//
// 【为什么必须切块（chunk）】
//   一本小说几十万字，塞不进一次请求（模型有输入上限），
//   也不能把整本压成一条向量 —— 那样"整本书的语义"会糊成一团，
//   检索只能定位到"某本书"，无法定位到"某一段"。
//   切成 500 字一块，每块单独算向量，检索才能精确定位。
// ============================================================

import "dotenv/config";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// adm-zip 用来解压 epub（epub 就是改了后缀的 zip）
import AdmZip from "adm-zip";
// cheerio 用 jQuery 风格的语法解析 xhtml，方便把正文从标签里剥出来
import { load } from "cheerio";

import {
  MilvusClient,
  DataType,
  IndexType,
  MetricType,
} from "@zilliz/milvus2-sdk-node";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { embeddings } from "./models.mjs";

// ---------- 配置 ----------

const COLLECTION_NAME = process.env.EBOOK_COLLECTION ?? "ebook_chunks";

// 必须和 .env 里 EMBEDDING_MODEL 的实际输出维度一致
const VECTOR_DIM = 1024;

// 每块多少字、相邻块重叠多少字
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 50;

// 每批发给 embedding 接口多少条。太大易触发限流，太小请求数太多
const EMBED_BATCH = 16;

// 每批写入 Milvus 多少行
const INSERT_BATCH = 50;

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// 路径优先级：命令行参数 > 环境变量 EBOOK_PATH > data/ebook.txt
const DEFAULT_EBOOK_PATH = path.resolve(__dirname, "../data/ebook.txt");
const EBOOK_PATH = process.argv[2]
  ? path.resolve(process.argv[2])
  : process.env.EBOOK_PATH
    ? path.resolve(process.env.EBOOK_PATH)
    : DEFAULT_EBOOK_PATH;

const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS ?? "localhost:19530",
});

// ---------- 内置演示文本 ----------
// 没有电子书时用来跑通流程，不是原著原文。
const DEMO_TEXT = `段誉是大理国镇南王段正淳之子，从小不喜习武，只爱读书，被父亲逼着练剑时常偷懒。
他因不愿学武离家出走，误入无量山，在琅嬛玉洞中见到了北冥神功与凌波微步两门绝学的图谱。
北冥神功能吸取他人内力化为己用，段誉起初不肯修炼，觉得这是损人利己的功夫，只在危急关头无意中使出。
凌波微步是一门轻功，按易经六十四卦的方位行走，脚下位置全凭卦象推算，身形飘忽不定，敌人极难近身。
后来段誉在天龙寺被鸠摩智擒住，辗转途中又习得六脉神剑。
六脉神剑以手指发出无形剑气，分为少商、商阳、中冲、关冲、少泽、少冲六路剑法，以内力催动，锋利无比。
段誉的六脉神剑时灵时不灵，因为他不愿与人争斗，内力运转不畅。
在少室山一战中，段誉终于以内力催动六脉神剑，击败了慕容复。

乔峰是丐帮帮主，武功以刚猛著称，最出名的是降龙十八掌，掌力雄浑。
他还会打狗棒法，这是丐帮帮主的信物绝学，招式以巧取胜，专攻敌人下盘。
乔峰身世坎坷，后来得知自己是契丹人，被迫离开丐帮，改名萧峰。

虚竹本是少林寺的小和尚，误打误撞破了珍珑棋局，被逍遥派掌门无崖子收为弟子。
之后虚竹成为灵鹫宫主人，学会了天山六阳掌、天山折梅手和生死符。`;

// ============================================================
// EPUB 解析
// ============================================================

/**
 * 从 zip 里读一个文件的文本。
 */
function readEntryText(zip, entryName) {
  const entry = zip.getEntry(entryName);
  if (!entry) {
    throw new Error(`epub 内找不到文件: ${entryName}`);
  }
  return entry.getData().toString("utf8");
}

/**
 * 把一章 xhtml 剥成纯文本。
 *
 * 做法：解析 HTML → 删掉 script/style → 取 h1 当章节标题 →
 * 把每个 <p> 的文本作为一段拼起来。
 * 不用 $("body").text() 是因为那样会把所有段落粘成一坨，没有换行。
 */
function extractChapterText(html) {
  const $ = load(html);
  $("script, style, head").remove();

  const heading = $("h1").first().text().trim();

  // 常规情况：正文都在 <p> 里
  let paragraphs = $("p")
    .map((_, el) => $(el).text().trim())
    .get()
    .filter((text) => text.length > 0);

  // 兜底：有些书用 <div> 排版，没有 <p>，那就取整个 body 再按行拆
  if (paragraphs.length === 0) {
    paragraphs = $("body")
      .text()
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
  }

  return {
    title: heading,
    text: paragraphs.join("\n"),
  };
}

/**
 * 解析 epub，返回按阅读顺序排列的章节数组。
 *
 * 为什么要读 OPF：epub 里的 xhtml 文件名不一定等于阅读顺序
 * （ch010.xhtml 可能是第一章）。OPF 里的 spine 才是权威顺序。
 */
async function parseEpub(filePath) {
  const zip = new AdmZip(filePath);

  // 1. container.xml 指向 OPF（书的信息清单）
  const containerXml = readEntryText(zip, "META-INF/container.xml");
  const opfPath = containerXml.match(/full-path="([^"]+)"/)?.[1];
  if (!opfPath) {
    throw new Error("epub 的 container.xml 里找不到 OPF 路径");
  }

  const opfDir = path.posix.dirname(opfPath);
  const opf = readEntryText(zip, opfPath);

  // 2. 书名和作者（取不到就用文件名兜底）
  const bookTitle =
    opf.match(/<dc:title[^>]*>([^<]*)<\/dc:title>/)?.[1]?.trim() ||
    path.basename(filePath, path.extname(filePath));
  const author =
    opf.match(/<dc:creator[^>]*>([^<]*)<\/dc:creator>/)?.[1]?.trim() || "";

  // 3. manifest：id -> href
  const manifest = new Map();
  const itemRe = /<item\s[^>]*\/>/g;
  for (const tag of opf.match(itemRe) ?? []) {
    const id = tag.match(/\bid="([^"]+)"/)?.[1];
    const href = tag.match(/\bhref="([^"]+)"/)?.[1];
    const mediaType = tag.match(/\bmedia-type="([^"]+)"/)?.[1] ?? "";
    if (id && href) {
      manifest.set(id, { href, mediaType });
    }
  }

  // 4. spine：按顺序列出 idref
  const spineBlock = opf.match(/<spine[^>]*>([\s\S]*?)<\/spine>/)?.[1] ?? "";
  const idrefs = [...spineBlock.matchAll(/<itemref[^>]*\bidref="([^"]+)"/g)].map(
    (m) => m[1],
  );

  // 5. 逐章提取
  const chapters = [];
  for (const idref of idrefs) {
    const item = manifest.get(idref);
    if (!item) continue;
    if (!/xhtml|html/.test(item.mediaType)) continue;

    const entryName = path.posix.join(opfDir, item.href);
    const html = readEntryText(zip, entryName);
    const { title, text } = extractChapterText(html);

    // 跳过没有正文的文件（比如封面页、目录页）
    if (text.length < 20) continue;

    chapters.push({
      num: chapters.length + 1,
      title: title || `第${chapters.length + 1}章`,
      text,
    });
  }

  return { title: bookTitle, author, chapters };
}

/**
 * 读取输入文件，统一返回 { title, author, chapters }。
 * 文件不存在时回落到演示文本。
 */
async function loadBook() {
  try {
    await fs.access(EBOOK_PATH);
  } catch {
    console.warn(`⚠ 没找到文件: ${EBOOK_PATH}`);
    console.warn("  改用内置演示文本跑通流程\n");
    return {
      title: "内置演示文本",
      author: "",
      chapters: [{ num: 1, title: "演示内容", text: DEMO_TEXT }],
      isDemo: true,
    };
  }

  const ext = path.extname(EBOOK_PATH).toLowerCase();

  if (ext === ".epub") {
    const parsed = await parseEpub(EBOOK_PATH);
    return { ...parsed, isDemo: false };
  }

  // 其他格式一律当纯文本，整本视作一章
  const text = await fs.readFile(EBOOK_PATH, "utf8");
  return {
    title: path.basename(EBOOK_PATH, ext),
    author: "",
    chapters: [{ num: 1, title: "全文", text }],
    isDemo: false,
  };
}

// ============================================================
// 切块与写入
// ============================================================

/**
 * 逐章切块。
 *
 * 【为什么按章切而不是先拼成一整本再切】
 * 那样切出来的块会跨章 —— 一块里前半段是第 3 章结尾、后半段是第 4 章开头，
 * 检索命中时无法判断它属于哪一章，也没法按章节过滤。
 * 逐章切能让每个块都带着准确的 chapter_num。
 */
async function splitChapters(chapters) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    // 【关键】默认分隔符是英文标点，中文小说几乎不会命中，
    // 结果会按固定字数硬切，经常把句子拦腰截断。
    // 这里按中文标点重排优先级。
    separators: ["\n\n", "\n", "。", "！", "？", "；", "，", ""],
  });

  const chunks = [];
  for (const chapter of chapters) {
    const docs = await splitter.createDocuments([chapter.text]);
    let indexInChapter = 0;

    for (const doc of docs) {
      const content = doc.pageContent.trim();
      if (!content) continue;

      chunks.push({
        content,
        chapterNum: chapter.num,
        chapterTitle: chapter.title,
        indexInChapter,
      });
      indexInChapter += 1;
    }
  }

  return chunks;
}

/**
 * 分批向量化。一次性丢几百块给接口会超时或被限流。
 */
async function embedInBatches(texts) {
  const vectors = [];

  for (let i = 0; i < texts.length; i += EMBED_BATCH) {
    const batch = texts.slice(i, i + EMBED_BATCH);

    // 免费/低配 key 常有速率限制，失败时等 1 秒重试一次
    let batchVectors;
    try {
      batchVectors = await embeddings.embedDocuments(batch);
    } catch (error) {
      console.warn(`  第 ${i / EMBED_BATCH + 1} 批失败（${error.message.slice(0, 60)}），1 秒后重试`);
      await new Promise((resolve) => setTimeout(resolve, 1000));
      batchVectors = await embeddings.embedDocuments(batch);
    }

    vectors.push(...batchVectors);
    console.log(`  向量化 ${vectors.length}/${texts.length}`);
  }

  return vectors;
}

async function main() {
  // ---------- 第 1 步：读文件 ----------
  const { title, author, chapters, isDemo } = await loadBook();

  console.log(`书名: ${title}${author ? `（${author}）` : ""}`);
  console.log(`来源: ${isDemo ? "内置演示文本" : EBOOK_PATH}`);
  console.log(`章节: ${chapters.length} 章`);
  console.log(`总字数: ${chapters.reduce((sum, c) => sum + c.text.length, 0)}`);

  // ---------- 第 2 步：逐章切块 ----------
  console.log(`\n开始切块（每块约 ${CHUNK_SIZE} 字，重叠 ${CHUNK_OVERLAP} 字）...`);
  const chunks = await splitChapters(chapters);
  console.log(`切出 ${chunks.length} 块`);
  console.log(`第一块预览: ${chunks[0].content.slice(0, 60)}...`);

  // ---------- 第 3 步：连库、重建集合 ----------
  await client.connectPromise;
  console.log(`\n已连接 Milvus: ${process.env.MILVUS_ADDRESS ?? "localhost:19530"}`);

  const { value: exists } = await client.hasCollection({
    collection_name: COLLECTION_NAME,
  });
  if (exists) {
    await client.dropCollection({ collection_name: COLLECTION_NAME });
    console.log(`集合 ${COLLECTION_NAME} 已存在，已删除并重建`);
  }

  await client.createCollection({
    collection_name: COLLECTION_NAME,
    fields: [
      { name: "id", data_type: DataType.VarChar, max_length: 64, is_primary_key: true },
      { name: "vector", data_type: DataType.FloatVector, dim: VECTOR_DIM },
      // 一块正文，500 字左右，5000 上限很宽松
      { name: "content", data_type: DataType.VarChar, max_length: 5000 },
      // 书名，便于同一集合放多本书
      { name: "book", data_type: DataType.VarChar, max_length: 200 },
      // 第几章，检索出来可以按章过滤或还原上下文
      { name: "chapter_num", data_type: DataType.Int64 },
      // 这一块在本章内的序号
      { name: "chunk_index", data_type: DataType.Int64 },
    ],
  });
  console.log(`集合 ${COLLECTION_NAME} 已创建`);

  // 几万块以内用 FLAT 精确检索即可；IVF_FLAT 需要额外调 nlist，小数据量反而更差
  await client.createIndex({
    collection_name: COLLECTION_NAME,
    field_name: "vector",
    index_type: IndexType.FLAT,
    metric_type: MetricType.COSINE,
  });
  await client.loadCollection({ collection_name: COLLECTION_NAME });
  console.log("索引已创建，集合已加载");

  // ---------- 第 4 步：分批向量化 ----------
  console.log("\n开始向量化...");
  const vectors = await embedInBatches(chunks.map((c) => c.content));

  const actualDim = vectors[0]?.length;
  if (actualDim !== VECTOR_DIM) {
    throw new Error(
      `向量维度不匹配：模型返回 ${actualDim} 维，集合定义 ${VECTOR_DIM} 维`,
    );
  }

  // ---------- 第 5 步：分批写入 ----------
  const rows = chunks.map((chunk, index) => ({
    // id 用「章号_块号」拼，同一本书内唯一
    id: `c${String(chunk.chapterNum).padStart(4, "0")}_${String(chunk.indexInChapter).padStart(3, "0")}`,
    vector: vectors[index],
    content: chunk.content,
    book: title,
    chapter_num: chunk.chapterNum,
    chunk_index: chunk.indexInChapter,
  }));

  console.log(`\n开始写入 Milvus（每批 ${INSERT_BATCH} 条）...`);
  let inserted = 0;
  for (let i = 0; i < rows.length; i += INSERT_BATCH) {
    const batch = rows.slice(i, i + INSERT_BATCH);
    const result = await client.insert({
      collection_name: COLLECTION_NAME,
      data: batch,
    });
    inserted += Number(result.insert_cnt);
    console.log(`  已写入 ${inserted}/${rows.length}`);
  }

  // 不 flush 的话，刚写入的数据可能立刻查不到
  await client.flush({ collection_names: [COLLECTION_NAME] });

  console.log(`\n完成：${inserted} 块已写入 ${COLLECTION_NAME}`);

  if (isDemo) {
    console.log("\n⚠ 当前用的是内置演示文本，检索效果有限。");
  }
}

main()
  .catch((error) => {
    console.error("\n失败:", error.message);
    process.exitCode = 1;
  })
  .finally(async () => {
    await client.closeConnection();
  });
