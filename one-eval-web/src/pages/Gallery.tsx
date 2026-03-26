import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  BookOpen,
  Brain,
  Code2,
  GraduationCap,
  ShieldCheck,
  Search,
  SlidersHorizontal,
  Tag,
  X,
  ExternalLink,
  RefreshCw,
  Loader2,
  Plus,
  Bot,
  MessageSquare,
  FlaskConical,
  FileSearch,
  Trash2,
  Upload,
  Eye,
  EyeOff,
  Table,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { useLang } from "@/lib/i18n";

// ============================================================================
// Types - 匹配 bench_gallery.json 的数据结构
// ============================================================================

type BenchCategory = "Math" | "Reasoning" | "Knowledge & QA" | "Safety & Alignment" | "Coding" | "Agents & Tools" | "Instruction & Chat" | "Long Context & RAG" | "Domain-Specific";

// bench_gallery.json 中的 HF 元数据
type HfMeta = {
  bench_name: string;
  hf_repo: string;
  card_text: string;
  tags: string[];
  exists_on_hf: boolean;
};

// 数据集结构信息
type DatasetStructure = {
  repo_id: string;
  revision: string | null;
  subsets: Array<{
    subset: string;
    splits: Array<{
      name: string;
      num_examples: number | null;
    }>;
    features: Record<string, unknown> | null;
  }>;
  ok: boolean;
  error: string | null;
};

// 下载配置
type DownloadConfig = {
  config: string;
  split: string;
  reason: string;
};

// 字段映射
type KeyMapping = {
  input_question_key: string | null;
  input_target_key: string | null;
  input_context_key?: string | null;
};

// bench_gallery.json 中的 meta 字段
type BenchGalleryMeta = {
  bench_name: string;
  source: string;
  aliases: string[];
  category: BenchCategory;
  tags: string[];
  description: string;
  description_zh?: string;
  hf_meta: HfMeta;
  structure?: DatasetStructure;
  download_config?: DownloadConfig;
  key_mapping?: KeyMapping;
  key_mapping_reason?: string;
  created_at?: string;  // 新增：创建时间
  artifact_paths?: string[];  // 附件路径列表
};

// bench_gallery.json 中的单个 bench 项
type BenchGalleryItem = {
  bench_name: string;
  bench_table_exist: boolean;
  bench_source_url: string;
  bench_dataflow_eval_type: string;
  bench_prompt_template: string | null;
  bench_keys: string[];
  meta: BenchGalleryMeta;
};

// 前端使用的简化类型（兼容旧代码）
type BenchItem = {
  id: string;
  name: string;
  meta: {
    category: BenchCategory;
    tags: string[];
    description: string;
    description_zh?: string;
    datasetUrl?: string;
    datasetKeys?: string[];
    source?: string;
    createdAt?: string;
    hasAttachment?: boolean;
  };
  // 保留完整的原始数据
  _raw?: BenchGalleryItem;
};

// ============================================================================
// Constants
// ============================================================================

const CATEGORIES: Array<{ id: BenchCategory | "All" }> = [
  { id: "All" },
  { id: "Knowledge & QA" },
  { id: "Reasoning" },
  { id: "Math" },
  { id: "Coding" },
  { id: "Long Context & RAG" },
  { id: "Instruction & Chat" },
  { id: "Agents & Tools" },
  { id: "Safety & Alignment" },
  { id: "Domain-Specific" },
];

// 排序选项
type SortOption = "newest" | "name_asc" | "name_desc" | "attachment_first";

const SORT_OPTIONS: Array<{ id: SortOption; label: { zh: string; en: string } }> = [
  { id: "newest", label: { zh: "最新创建", en: "Newest First" } },
  { id: "attachment_first", label: { zh: "有附件优先", en: "With Attachments First" } },
  { id: "name_asc", label: { zh: "名称 A-Z", en: "Name A-Z" } },
  { id: "name_desc", label: { zh: "名称 Z-A", en: "Name Z-A" } },
];

// ============================================================================
// Utility Functions
// ============================================================================

function getBenchIcon(category: BenchCategory) {
  switch (category) {
    case "Math":
      return { Icon: BookOpen, bg: "bg-emerald-50", fg: "text-emerald-600" };
    case "Reasoning":
      return { Icon: Brain, bg: "bg-indigo-50", fg: "text-indigo-600" };
    case "Knowledge & QA":
      return { Icon: GraduationCap, bg: "bg-sky-50", fg: "text-sky-600" };
    case "Safety & Alignment":
      return { Icon: ShieldCheck, bg: "bg-amber-50", fg: "text-amber-700" };
    case "Coding":
      return { Icon: Code2, bg: "bg-violet-50", fg: "text-violet-600" };
    case "Agents & Tools":
      return { Icon: Bot, bg: "bg-rose-50", fg: "text-rose-600" };
    case "Instruction & Chat":
      return { Icon: MessageSquare, bg: "bg-pink-50", fg: "text-pink-600" };
    case "Long Context & RAG":
      return { Icon: FileSearch, bg: "bg-cyan-50", fg: "text-cyan-600" };
    case "Domain-Specific":
      return { Icon: FlaskConical, bg: "bg-orange-50", fg: "text-orange-600" };
    default:
      return { Icon: Tag, bg: "bg-slate-50", fg: "text-slate-600" };
  }
}

/**
 * 将 bench_gallery.json 的数据转换为前端使用的 BenchItem 格式
 */
function transformBenchGalleryItem(item: BenchGalleryItem): BenchItem {
  const meta = item.meta || {};
  // 从 aliases 中获取显示名称（通常第二个是大写版本）
  const displayName = meta.aliases?.[1] || meta.aliases?.[0] || item.bench_name;

  // 判断是否有附件（本地上传的数据集或有 artifact_paths）
  const hasAttachment = meta.source === "local_upload" ||
    (meta.artifact_paths && meta.artifact_paths.length > 0) ||
    item.bench_source_url?.startsWith("local://");

  return {
    id: item.bench_name,
    name: displayName,
    meta: {
      category: meta.category || "Knowledge & QA",
      tags: meta.tags || [],
      description: meta.description || "",
      description_zh: meta.description_zh || "",
      datasetUrl: item.bench_source_url,
      datasetKeys: item.bench_keys,
      source: meta.source,
      createdAt: meta.created_at,
      hasAttachment,
    },
    _raw: item,
  };
}

function getApiBaseUrl(): string {
  return localStorage.getItem("oneEval.apiBaseUrl") || "http://localhost:8000";
}

function loadGalleryBenches(): BenchItem[] {
  try {
    const raw = localStorage.getItem("oneEval.gallery.benches");
    if (!raw) return [];
    const parsed = JSON.parse(raw) as BenchItem[];
    if (!Array.isArray(parsed) || parsed.length === 0) return [];
    return parsed;
  } catch {
    return [];
  }
}

function saveGalleryBenches(items: BenchItem[]) {
  localStorage.setItem("oneEval.gallery.benches", JSON.stringify(items));
}

// ============================================================================
// Component
// ============================================================================

// Bench 类型选项（对应新分类）
const BENCH_TYPES = [
  "knowledge",
  "language & reasoning",
  "math",
  "coding",
  "information retrieval & RAG",
  "instruction-following",
  "conversation & chatbots",
  "agents & tools use",
  "safety",
  "bias & ethics",
  "domain-specific",
  "multilingual",
  "other",
];

// 有效的评测数据流类型（对应后端 _VALID_EVAL_TYPES）
const EVAL_TYPES = [
  { value: "key1_text_score", label: { zh: "文本评分 (Text Score)", en: "Text Score" } },
  { value: "key2_qa", label: { zh: "问答 (QA)", en: "QA" } },
  { value: "key2_q_ma", label: { zh: "多答案问答 (Multi-Answer QA)", en: "Multi-Answer QA" } },
  { value: "key3_q_choices_a", label: { zh: "单选题 (Single Choice)", en: "Single Choice" } },
  { value: "key3_q_choices_as", label: { zh: "多选题 (Multiple Choice)", en: "Multiple Choice" } },
  { value: "key3_q_a_rejected", label: { zh: "拒绝评测 (Rejected)", en: "Rejected" } },
  { value: "other", label: { zh: "其他/自定义 (Other)", en: "Other" } },
];

function evalTypeLabel(value: string, lang: string) {
  const item = EVAL_TYPES.find(t => t.value === value);
  if (!item) return value;
  return lang === "zh" ? item.label.zh : item.label.en;
}

export const Gallery = () => {
  const { lang } = useLang();
  const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
  const categoryLabel = (id: BenchCategory | "All" | string) => {
    const map: Record<string, { zh: string; en: string }> = {
      All: { zh: "全部", en: "All" },
      "Knowledge & QA": { zh: "知识问答", en: "Knowledge & QA" },
      Reasoning: { zh: "推理", en: "Reasoning" },
      Math: { zh: "数学", en: "Math" },
      Coding: { zh: "编程", en: "Coding" },
      "Long Context & RAG": { zh: "长上下文与 RAG", en: "Long Context & RAG" },
      "Instruction & Chat": { zh: "指令与对话", en: "Instruction & Chat" },
      "Agents & Tools": { zh: "Agent 与工具", en: "Agents & Tools" },
      "Safety & Alignment": { zh: "安全与对齐", en: "Safety & Alignment" },
      "Domain-Specific": { zh: "领域专项", en: "Domain-Specific" },
      General: { zh: "通用", en: "General" },
    };
    const item = map[id] || map["General"];
    return tt(item.zh, item.en);
  };
  const benchTypeLabel = (value: string) => {
    const map: Record<string, { zh: string; en: string }> = {
      knowledge: { zh: "知识", en: "knowledge" },
      "language & reasoning": { zh: "语言与推理", en: "language & reasoning" },
      math: { zh: "数学", en: "math" },
      coding: { zh: "编程", en: "coding" },
      "information retrieval & RAG": { zh: "信息检索与 RAG", en: "information retrieval & RAG" },
      "instruction-following": { zh: "指令跟随", en: "instruction-following" },
      "conversation & chatbots": { zh: "对话与聊天机器人", en: "conversation & chatbots" },
      "agents & tools use": { zh: "Agent 与工具使用", en: "agents & tools use" },
      safety: { zh: "安全", en: "safety" },
      "bias & ethics": { zh: "偏见与伦理", en: "bias & ethics" },
      "domain-specific": { zh: "领域专项", en: "domain-specific" },
      multilingual: { zh: "多语言", en: "multilingual" },
      other: { zh: "其他", en: "other" },
    };
    const hit = map[value];
    return hit ? tt(hit.zh, hit.en) : value;
  };
  const navigate = useNavigate();
  const [benches, setBenches] = useState<BenchItem[]>([]);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<(typeof CATEGORIES)[number]["id"]>("All");
  const [activeBenchId, setActiveBenchId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<SortOption>("attachment_first");  // 默认有附件优先
  const [showOnlyUploaded, setShowOnlyUploaded] = useState(false);  // 只显示本地上传的

  // Add Bench Modal 状态
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [addForm, setAddForm] = useState({
    bench_name: "",
    type: "knowledge",
    description: "",
    dataset_url: "",
    eval_type: "key2_qa",  // 评测数据流类型
  });
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadMode, setUploadMode] = useState<"url" | "file">("url");

  // 文件预览状态
  const [previewData, setPreviewData] = useState<{
    columns: string[];
    rows: Record<string, unknown>[];
    format: string;
  } | null>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [fieldAnalysis, setFieldAnalysis] = useState<{
    fields: string[];
    suggestions: {
      prompt_fields: Array<{ field: string; reason: string }>;
      target_fields: Array<{ field: string; reason: string }>;
      context_fields: Array<{ field: string; reason: string }>;
      field_types: Record<string, { type: string; avg_length?: number; sample?: unknown; keys?: string[] }>;
    };
    sample_rows: Record<string, unknown>[];
  } | null>(null);

  // 追加附件状态
  const [isAppendModalOpen, setIsAppendModalOpen] = useState(false);
  const [appendFile, setAppendFile] = useState<File | null>(null);
  const [isAppending, setIsAppending] = useState(false);

  // 数据预览弹窗状态
  const [isPreviewModalOpen, setIsPreviewModalOpen] = useState(false);
  const [showAllData, setShowAllData] = useState(false);  // 是否显示全部数据
  const [totalRows, setTotalRows] = useState<number | null>(null);  // 总行数

  // 侧边栏宽度状态
  const [sidebarWidth, setSidebarWidth] = useState(420);
  const [isResizing, setIsResizing] = useState(false);

  // 从 API 获取 bench 数据
  const fetchBenches = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const apiBaseUrl = getApiBaseUrl();
      const response = await fetch(`${apiBaseUrl}/api/benches/gallery`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: BenchGalleryItem[] = await response.json();
      const transformed = data.map(transformBenchGalleryItem);
      setBenches(transformed);
      saveGalleryBenches(transformed);
    } catch (err) {
      console.error("Failed to fetch benches:", err);
      setError(err instanceof Error ? err.message : tt("获取 Bench 失败", "Failed to fetch benches"));
      // 尝试从 localStorage 加载缓存数据
      const cached = loadGalleryBenches();
      if (cached.length > 0) {
        setBenches(cached);
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // 先尝试从缓存加载，然后从 API 刷新
    const cached = loadGalleryBenches();
    if (cached.length > 0) {
      setBenches(cached);
      setIsLoading(false);
    }
    fetchBenches();
  }, []);

  // 打开侧边栏时锁定页面滚动
  useEffect(() => {
    if (activeBenchId) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [activeBenchId]);

  // 侧边栏拖拽调整宽度
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = window.innerWidth - e.clientX;
      const clampedWidth = Math.max(360, Math.min(800, newWidth));
      setSidebarWidth(clampedWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing]);

  const activeBench = useMemo(() => benches.find((b) => b.id === activeBenchId) ?? null, [benches, activeBenchId]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    let result = benches
      .filter((b) => (category === "All" ? true : b.meta.category === category))
      .filter((b) => {
        if (!q) return true;
        const hay = `${b.name} ${b.id} ${b.meta.description} ${b.meta.description_zh || ""} ${b.meta.tags.join(" ")} ${b.meta.category}`.toLowerCase();
        return hay.includes(q);
      })
      .filter((b) => {
        // 只显示本地上传的（有附件的）
        if (showOnlyUploaded) {
          return b.meta.hasAttachment || b.meta.source === "local_upload";
        }
        return true;
      });

    // 排序
    result = result.sort((a, b) => {
      switch (sortBy) {
        case "newest": {
          // 有创建时间的靠前，然后按时间倒序
          const aTime = a.meta.createdAt ? new Date(a.meta.createdAt).getTime() : 0;
          const bTime = b.meta.createdAt ? new Date(b.meta.createdAt).getTime() : 0;
          // 新建的（有时间）排前面
          if (aTime && !bTime) return -1;
          if (!aTime && bTime) return 1;
          return bTime - aTime;
        }
        case "attachment_first":
          // 有附件的靠前
          if (a.meta.hasAttachment && !b.meta.hasAttachment) return -1;
          if (!a.meta.hasAttachment && b.meta.hasAttachment) return 1;
          // 其次按名称
          return a.name.localeCompare(b.name);
        case "name_asc":
          return a.name.localeCompare(b.name);
        case "name_desc":
          return b.name.localeCompare(a.name);
        default:
          return 0;
      }
    });

    return result;
  }, [benches, query, category, sortBy, showOnlyUploaded]);

  const handleUseBench = (benchId: string) => {
    navigate("/eval", { state: { preSelectedBench: benchId } });
  };

  const handleUpdateBench = (updated: BenchItem) => {
    setBenches((prev) => {
      const next = prev.map((b) => (b.id === updated.id ? updated : b));
      saveGalleryBenches(next);
      return next;
    });
  };

  const handleRefresh = () => {
    fetchBenches();
  };

  const handleDeleteBench = async (benchId: string) => {
    if (!window.confirm(tt("确定要删除这个 Benchmark 吗？此操作不可撤销。", "Are you sure you want to delete this benchmark? This action cannot be undone."))) {
      return;
    }

    try {
      const apiBaseUrl = getApiBaseUrl();
      const response = await fetch(`${apiBaseUrl}/api/benches/gallery/${encodeURIComponent(benchId)}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || tt("删除失败", "Delete failed"));
      }

      // 从本地状态中移除
      setBenches((prev) => {
        const next = prev.filter((b) => b.id !== benchId);
        saveGalleryBenches(next);
        return next;
      });
      setActiveBenchId(null);
    } catch (err) {
      alert(err instanceof Error ? err.message : tt("删除失败", "Delete failed"));
    }
  };

  const handlePreviewBench = async (benchId: string, loadAll: boolean = false) => {
    setIsPreviewModalOpen(true);
    setIsPreviewLoading(true);
    if (!loadAll) {
      setPreviewData(null);
      setFieldAnalysis(null);
    }
    setShowAllData(loadAll);
    try {
      const apiBaseUrl = getApiBaseUrl();

      if (loadAll) {
        // 加载全部数据 (limit=0 表示不限制)
        const previewRes = await fetch(`${apiBaseUrl}/api/benches/preview/${encodeURIComponent(benchId)}?limit=0`);
        if (previewRes.ok) {
          const data = await previewRes.json();
          setPreviewData({
            columns: data.columns || [],
            rows: data.rows || [],
            format: data.format || "unknown",
          });
          setTotalRows(data.rows?.length || 0);
        }
      } else {
        // 并行获取预览（默认 10 条）和字段分析
        const [previewRes, analysisRes] = await Promise.all([
          fetch(`${apiBaseUrl}/api/benches/preview/${encodeURIComponent(benchId)}`),
          fetch(`${apiBaseUrl}/api/benches/analyze_fields/${encodeURIComponent(benchId)}`),
        ]);

        if (previewRes.ok) {
          const data = await previewRes.json();
          setPreviewData({
            columns: data.columns || [],
            rows: data.rows || [],
            format: data.format || "unknown",
          });
          setTotalRows(data.total_shown || data.rows?.length || 0);
        }

        if (analysisRes.ok) {
          const analysis = await analysisRes.json();
          setFieldAnalysis(analysis);
        }
      }
    } catch (err) {
      console.error("Failed to preview bench:", err);
    } finally {
      setIsPreviewLoading(false);
    }
  };

  const handleAppendAttachment = async () => {
    if (!activeBench || !appendFile) return;

    setIsAppending(true);
    try {
      const apiBaseUrl = getApiBaseUrl();
      const formData = new FormData();
      formData.append("file", appendFile);
      formData.append("bench_name", activeBench.id);

      const response = await fetch(`${apiBaseUrl}/api/benches/append_attachment`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || tt("追加失败", "Append failed"));
      }

      // 刷新 bench 数据
      await fetchBenches();
      setIsAppendModalOpen(false);
      setAppendFile(null);
      alert(tt("附件追加成功！", "Attachment appended successfully!"));
    } catch (err) {
      alert(err instanceof Error ? err.message : tt("追加失败", "Append failed"));
    } finally {
      setIsAppending(false);
    }
  };

  const handleAddBench = async () => {
    if (!addForm.bench_name.trim() || !addForm.description.trim()) {
      return;
    }

    // 文件上传模式需要选择文件
    if (uploadMode === "file" && !uploadFile) {
      alert(tt("请选择要上传的文件", "Please select a file to upload"));
      return;
    }

    setIsSubmitting(true);
    try {
      const apiBaseUrl = getApiBaseUrl();

      if (uploadMode === "file" && uploadFile) {
        // 文件上传模式
        const formData = new FormData();
        formData.append("file", uploadFile);
        formData.append("bench_name", addForm.bench_name.trim());
        formData.append("eval_type", addForm.eval_type || "key2_qa");  // 使用 eval_type 字段
        formData.append("description", addForm.description.trim());

        const response = await fetch(`${apiBaseUrl}/api/benches/upload`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || tt("上传失败", "Upload failed"));
        }

        await fetchBenches();
        setIsAddModalOpen(false);
        setAddForm({ bench_name: "", type: "knowledge", description: "", dataset_url: "", eval_type: "key2_qa" });
        setUploadFile(null);
        setUploadMode("url");
      } else {
        // URL 模式
        const response = await fetch(`${apiBaseUrl}/api/benches/gallery`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            bench_name: addForm.bench_name.trim(),
            type: addForm.type,
            description: addForm.description.trim(),
            dataset_url: addForm.dataset_url.trim() || null,
          }),
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || tt("新增 Bench 失败", "Failed to add bench"));
        }

        // 成功后刷新列表并关闭弹窗
        await fetchBenches();
        setIsAddModalOpen(false);
        setAddForm({ bench_name: "", type: "knowledge", description: "", dataset_url: "", eval_type: "key2_qa" });
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : tt("新增 Bench 失败", "Failed to add bench"));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="p-12 max-w-7xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-4xl font-bold tracking-tight text-slate-900">{tt("基准库", "Benchmark Gallery")}</h1>
          <p className="text-slate-600 text-lg">{tt("搜索、筛选并配置你的精选基准。", "Search, filter, and configure your curated benchmarks.")}</p>
        </div>
        <div className="flex gap-3">
          <Button
            className="bg-gradient-to-r from-blue-600 to-violet-600 text-white hover:from-blue-500 hover:to-violet-500"
            onClick={() => setIsAddModalOpen(true)}
          >
            <Plus className="w-4 h-4 mr-2" />
            {tt("新增 Bench", "Add Bench")}
          </Button>
          <Button
            variant="outline"
            className="border-slate-200"
            onClick={handleRefresh}
            disabled={isLoading}
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4 mr-2" />
            )}
            {tt("刷新", "Refresh")}
          </Button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}. {tt("已展示缓存数据。", "Showing cached data.")}
        </div>
      )}

      <div className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row gap-3 md:items-center md:justify-between">
          <div className="relative w-full md:max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={tt("搜索基准、标签、分类...", "Search benches, tags, categories...")}
              className="pl-9 bg-white border-slate-200"
            />
          </div>
          <div className="flex items-center gap-3">
            {/* 排序选择 */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortOption)}
              className="h-9 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              title={tt("排序方式", "Sort by")}
              aria-label={tt("排序方式", "Sort by")}
            >
              {SORT_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {tt(opt.label.zh, opt.label.en)}
                </option>
              ))}
            </select>
            {/* 本地上传筛选 */}
            <button
              type="button"
              onClick={() => setShowOnlyUploaded(!showOnlyUploaded)}
              className={cn(
                "px-3 py-1.5 text-sm rounded-full border transition-colors flex items-center gap-1.5",
                showOnlyUploaded
                  ? "bg-emerald-600 text-white border-emerald-600"
                  : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
              )}
            >
              <Upload className="w-3.5 h-3.5" />
              {tt("本地上传评测集", "Uploaded Datasets")}
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {CATEGORIES.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setCategory(c.id)}
                  className={cn(
                    "px-3 py-1.5 text-sm rounded-full border transition-colors",
                    c.id === category
                  ? "bg-gradient-to-r from-blue-600 to-violet-600 text-white border-transparent shadow-sm shadow-blue-600/20"
                  : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                  )}
                >
                  {categoryLabel(c.id)}
                </button>
          ))}
        </div>
      </div>

      {isLoading && benches.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-slate-500">
          <Loader2 className="w-8 h-8 animate-spin mb-4" />
          <p>{tt("正在加载基准...", "Loading benchmarks...")}</p>
        </div>
      ) : benches.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-slate-500">
          <p>{tt("暂无可用基准。", "No benchmarks available.")}</p>
          <Button variant="outline" className="mt-4" onClick={handleRefresh}>
            {tt("重试", "Try Again")}
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filtered.map((bench, idx) => {
            const { Icon, bg, fg } = getBenchIcon(bench.meta.category);
            return (
              <motion.div key={bench.id} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.04 }}>
                <Card className="h-full flex flex-col border-slate-200 hover:shadow-lg transition-shadow duration-300">
                  <CardHeader>
                    <div className="flex justify-between items-start gap-4">
                      <div className="flex items-center gap-3">
                        <div className={cn("w-12 h-12 rounded-2xl flex items-center justify-center", bg)}>
                          <Icon className={cn("w-6 h-6", fg)} />
                        </div>
                        <div>
                          <CardTitle className="text-xl text-slate-900">{bench.name}</CardTitle>
                          <div className="text-xs text-slate-500 mt-0.5">{categoryLabel(bench.meta.category)}</div>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-2 mt-4">
                      {/* 本地上传标签优先显示 */}
                      {(bench.meta.hasAttachment || bench.meta.source === "local_upload") && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1">
                          <Upload className="w-3 h-3" />
                          {tt("本地上传", "Uploaded")}
                        </span>
                      )}
                      {bench.meta.tags.filter(tag => tag !== "本地上传评测集" && tag !== "uploaded").slice(0, 3).map((tag) => (
                        <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-slate-50 text-slate-600 border border-slate-200">
                          {tag}
                        </span>
                      ))}
                      {bench.meta.tags.length > 3 && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-slate-50 text-slate-500 border border-slate-200">
                          +{bench.meta.tags.length - 3}
                        </span>
                      )}
                    </div>
                  </CardHeader>

                  <CardContent className="flex-1">
                    <CardDescription className="text-sm text-slate-600 line-clamp-3">{lang === "zh" ? (bench.meta.description_zh || bench.meta.description) : bench.meta.description}</CardDescription>
                  </CardContent>

                  <CardFooter className="pt-4 border-t border-slate-100 bg-slate-50/30 flex gap-2">
                    <Button
                      className="flex-1 text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-sm shadow-blue-600/20"
                      onClick={() => handleUseBench(bench.id)}
                    >
                      {tt("使用", "Use")}
                    </Button>
                    {bench.meta.datasetUrl && (
                      <Button
                        variant="outline"
                        className="border-slate-200"
                        onClick={() => window.open(bench.meta.datasetUrl, "_blank")}
                      >
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    )}
                    <Button variant="outline" className="border-slate-200" onClick={() => setActiveBenchId(bench.id)}>
                      <SlidersHorizontal className="w-4 h-4" />
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            );
          })}
        </div>
      )}

      <AnimatePresence>
        {activeBench && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50"
          >
            <div className="absolute inset-0 bg-black/20" onClick={() => setActiveBenchId(null)} />
            <motion.div
              initial={{ x: sidebarWidth }}
              animate={{ x: 0 }}
              exit={{ x: sidebarWidth }}
              transition={{ type: "spring", stiffness: 280, damping: 30 }}
              className="absolute right-0 top-0 bottom-0 bg-white border-l border-slate-200 shadow-2xl flex"
              style={{ width: sidebarWidth }}
              role="dialog"
              aria-modal="true"
            >
              {/* 左侧拖拽调整宽度的把手 */}
              <div
                className="w-1 bg-transparent hover:bg-blue-400 cursor-col-resize transition-colors flex-shrink-0"
                onMouseDown={() => setIsResizing(true)}
              />
              {/* 内容区域 */}
              <div className="flex-1 overflow-y-auto p-6">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-xs text-slate-500 uppercase tracking-wider">{tt("配置 Bench", "Configure Bench")}</div>
                    <div className="text-2xl font-bold text-slate-900 mt-1">{activeBench.name}</div>
                  </div>
                  <button
                    className="p-2 rounded-lg hover:bg-slate-100 text-slate-500"
                    onClick={() => setActiveBenchId(null)}
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="mt-6 space-y-5">
                  <div className="space-y-2">
                    <Label>{tt("显示名称", "Display Name")}</Label>
                    <Input
                      value={activeBench.name}
                      onChange={(e) => handleUpdateBench({ ...activeBench, name: e.target.value })}
                      className="border-slate-200"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>{tt("描述", "Description")}</Label>
                    <textarea
                      value={activeBench.meta.description}
                      onChange={(e) =>
                        handleUpdateBench({ ...activeBench, meta: { ...activeBench.meta, description: e.target.value } })
                      }
                      className="w-full min-h-[120px] rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>{tt("分类", "Category")}</Label>
                    <select
                      value={activeBench.meta.category}
                      onChange={(e) =>
                        handleUpdateBench({
                          ...activeBench,
                          meta: { ...activeBench.meta, category: e.target.value as BenchCategory },
                        })
                      }
                      className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                    >
                      {CATEGORIES.filter((c) => c.id !== "All").map((c) => (
                        <option key={c.id} value={c.id}>
                          {categoryLabel(c.id as BenchCategory)}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-2">
                    <Label>{tt("标签（逗号分隔）", "Tags (comma-separated)")}</Label>
                    <Input
                      value={activeBench.meta.tags.join(", ")}
                      onChange={(e) =>
                        handleUpdateBench({
                          ...activeBench,
                          meta: {
                            ...activeBench.meta,
                            tags: e.target.value
                              .split(",")
                              .map((t) => t.trim())
                              .filter(Boolean),
                          },
                        })
                      }
                      className="border-slate-200"
                    />
                  </div>

                  {activeBench.meta.datasetUrl && (
                    <div className="space-y-2">
                      <Label>{tt("数据集链接", "Dataset URL")}</Label>
                      <div className="flex gap-2">
                        <Input
                          value={activeBench.meta.datasetUrl}
                          readOnly
                          className="border-slate-200 bg-slate-50 text-slate-600 flex-1"
                        />
                        <Button
                          variant="outline"
                          className="border-slate-200 shrink-0"
                          onClick={() => window.open(activeBench.meta.datasetUrl, "_blank")}
                        >
                          <ExternalLink className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  )}

                  {activeBench.meta.datasetKeys && activeBench.meta.datasetKeys.length > 0 && (
                    <div className="space-y-2">
                      <Label>{tt("数据字段 Keys", "Dataset Keys")}</Label>
                      <div className="flex flex-wrap gap-1.5 p-3 rounded-md border border-slate-200 bg-slate-50">
                        {activeBench.meta.datasetKeys.map((key) => (
                          <span key={key} className="text-xs px-2 py-1 rounded bg-white border border-slate-200 text-slate-600 font-mono">
                            {key}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* 显示额外的 bench_gallery.json 信息 */}
                  {activeBench._raw && (
                    <>
                      {activeBench._raw.meta?.download_config && (
                        <div className="space-y-2">
                          <Label>{tt("下载配置", "Download Config")}</Label>
                          <div className="p-3 rounded-md border border-slate-200 bg-slate-50 text-xs font-mono space-y-1">
                            <div><span className="text-slate-500">config:</span> {activeBench._raw.meta.download_config.config}</div>
                            <div><span className="text-slate-500">split:</span> {activeBench._raw.meta.download_config.split}</div>
                          </div>
                        </div>
                      )}

                      {activeBench._raw.meta?.key_mapping && (
                        <div className="space-y-2">
                          <Label>{tt("字段映射", "Key Mapping")}</Label>
                          <div className="p-3 rounded-md border border-slate-200 bg-slate-50 text-xs font-mono space-y-1">
                            {activeBench._raw.meta.key_mapping.input_question_key && (
                              <div><span className="text-slate-500">question:</span> {activeBench._raw.meta.key_mapping.input_question_key}</div>
                            )}
                            {activeBench._raw.meta.key_mapping.input_target_key && (
                              <div><span className="text-slate-500">target:</span> {activeBench._raw.meta.key_mapping.input_target_key}</div>
                            )}
                            {activeBench._raw.meta.key_mapping.input_context_key && (
                              <div><span className="text-slate-500">context:</span> {activeBench._raw.meta.key_mapping.input_context_key}</div>
                            )}
                          </div>
                        </div>
                      )}

                      {activeBench._raw.bench_dataflow_eval_type && (
                        <div className="space-y-2">
                          <Label>{tt("评测类型", "Eval Type")}</Label>
                          <div className="px-3 py-2 rounded-md border border-slate-200 bg-slate-50 text-sm text-slate-600">
                            {activeBench._raw.bench_dataflow_eval_type}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>

                <div className="mt-8 flex gap-2">
                  <Button className="flex-1 bg-slate-900 text-white hover:bg-slate-800" onClick={() => handleUseBench(activeBench.id)}>
                    {tt("使用该 Bench", "Use This Bench")}
                  </Button>
                  <Button
                    variant="outline"
                    className="border-slate-200"
                    onClick={() => handlePreviewBench(activeBench.id)}
                    title={tt("预览数据", "Preview Data")}
                  >
                    <Eye className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="outline"
                    className="border-slate-200"
                    onClick={() => setIsAppendModalOpen(true)}
                    title={tt("追加附件", "Append Attachment")}
                  >
                    <span className="sr-only">{tt("追加附件", "Append Attachment")}</span>
                    <Upload className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="outline"
                    className="border-red-200 text-red-600 hover:bg-red-50"
                    onClick={() => handleDeleteBench(activeBench.id)}
                    title={tt("删除", "Delete")}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                  <Button variant="outline" className="border-slate-200" onClick={() => setActiveBenchId(null)}>
                    {tt("关闭", "Close")}
                  </Button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 数据预览弹窗 */}
      <AnimatePresence>
        {isPreviewModalOpen && activeBench && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] flex items-center justify-center p-4"
          >
            <div className="absolute inset-0 bg-black/30" onClick={() => setIsPreviewModalOpen(false)} />
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="relative bg-white rounded-2xl shadow-2xl w-full max-w-6xl max-h-[90vh] flex flex-col"
            >
              {/* 弹窗头部 */}
              <div className="flex items-center justify-between p-4 border-b border-slate-200">
                <div className="flex items-center gap-3">
                  <Table className="w-5 h-5 text-slate-500" />
                  <div>
                    <h2 className="text-lg font-bold text-slate-900">{tt("数据预览", "Data Preview")}</h2>
                    <p className="text-sm text-slate-500">{activeBench.name}</p>
                  </div>
                </div>
                <button
                  className="p-2 rounded-lg hover:bg-slate-100 text-slate-500"
                  onClick={() => setIsPreviewModalOpen(false)}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* 弹窗内容 */}
              <div className="flex-1 overflow-auto p-4">
                {isPreviewLoading ? (
                  <div className="flex items-center justify-center py-20">
                    <Loader2 className="w-8 h-8 animate-spin mr-3" />
                    <span className="text-slate-600">{tt("加载预览中...", "Loading preview...")}</span>
                  </div>
                ) : previewData ? (
                  <div className="space-y-6">
                    {/* 字段识别结果 */}
                    {fieldAnalysis && (
                      <div className="space-y-3">
                        <h3 className="font-semibold text-slate-800">{tt("字段识别", "Field Detection")}</h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          {fieldAnalysis.suggestions.prompt_fields.length > 0 && (
                            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                              <div className="font-medium text-blue-800 mb-2 text-sm">{tt("Prompt 字段", "Prompt Fields")}</div>
                              <div className="flex flex-wrap gap-1">
                                {fieldAnalysis.suggestions.prompt_fields.map((f) => (
                                  <span key={f.field} className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-mono">
                                    {f.field}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          {fieldAnalysis.suggestions.target_fields.length > 0 && (
                            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                              <div className="font-medium text-green-800 mb-2 text-sm">{tt("Target 字段", "Target Fields")}</div>
                              <div className="flex flex-wrap gap-1">
                                {fieldAnalysis.suggestions.target_fields.map((f) => (
                                  <span key={f.field} className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs font-mono">
                                    {f.field}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          {fieldAnalysis.suggestions.context_fields.length > 0 && (
                            <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                              <div className="font-medium text-amber-800 mb-2 text-sm">{tt("Context 字段", "Context Fields")}</div>
                              <div className="flex flex-wrap gap-1">
                                {fieldAnalysis.suggestions.context_fields.map((f) => (
                                  <span key={f.field} className="px-2 py-0.5 bg-amber-100 text-amber-700 rounded text-xs font-mono">
                                    {f.field}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                        {/* 所有字段类型 */}
                        {Object.keys(fieldAnalysis.suggestions.field_types).length > 0 && (
                          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                            <div className="font-medium text-slate-700 mb-2 text-sm">{tt("所有字段类型", "All Field Types")}</div>
                            <div className="flex flex-wrap gap-2">
                              {Object.entries(fieldAnalysis.suggestions.field_types).map(([field, info]) => (
                                <div key={field} className="flex items-center gap-2 py-1 px-2 rounded bg-white border border-slate-200">
                                  <span className="font-mono text-xs text-slate-700">{field}</span>
                                  <span className={`text-xs px-1.5 py-0.5 rounded ${
                                    info.type === "string" ? "bg-blue-100 text-blue-700" :
                                    info.type === "list" ? "bg-purple-100 text-purple-700" :
                                    info.type === "number" ? "bg-green-100 text-green-700" :
                                    "bg-slate-100 text-slate-600"
                                  }`}>
                                    {info.type}
                                  </span>
                                  {info.avg_length && (
                                    <span className="text-xs text-slate-400">~{info.avg_length}</span>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* 数据表格 */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-slate-800">{tt("数据内容", "Data Content")}</h3>
                        <span className="text-xs px-2 py-1 bg-slate-100 text-slate-600 rounded">{previewData.format.toUpperCase()}</span>
                      </div>
                      <div className="border border-slate-200 rounded-lg overflow-hidden">
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead className="bg-slate-100 sticky top-0">
                              <tr>
                                {previewData.columns.map((col) => (
                                  <th key={col} className="px-4 py-3 text-left font-medium text-slate-700 border-b border-slate-200 whitespace-nowrap">
                                    {col}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {previewData.rows.map((row, idx) => (
                                <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                                  {previewData.columns.map((col) => (
                                    <td key={col} className="px-4 py-3 text-slate-600 max-w-[300px]">
                                      <div className="truncate" title={typeof row[col] === "object" ? JSON.stringify(row[col]) : String(row[col] || "")}>
                                        {typeof row[col] === "object"
                                          ? <span className="text-purple-600 font-mono text-xs">{JSON.stringify(row[col]).slice(0, 80)}...</span>
                                          : String(row[col] || "").slice(0, 200) || <span className="text-slate-300">-</span>}
                                      </div>
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="text-xs text-slate-500">
                          {showAllData
                            ? tt(`共 ${previewData.rows.length} 条数据`, `${previewData.rows.length} rows total`)
                            : tt(`显示前 ${previewData.rows.length} 条数据`, `Showing first ${previewData.rows.length} rows`)}
                        </div>
                        <div className="flex gap-2">
                          {!showAllData && previewData.rows.length === 10 && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handlePreviewBench(activeBench.id, true)}
                              disabled={isPreviewLoading}
                            >
                              {isPreviewLoading ? (
                                <><Loader2 className="w-3 h-3 mr-1 animate-spin" />{tt("加载中...", "Loading...")}</>
                              ) : (
                                <><Eye className="w-3 h-3 mr-1" />{tt("显示全部", "Show All")}</>
                              )}
                            </Button>
                          )}
                          {showAllData && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handlePreviewBench(activeBench.id, false)}
                              disabled={isPreviewLoading}
                            >
                              <><EyeOff className="w-3 h-3 mr-1" />{tt("收起", "Collapse")}</>
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center py-20 text-slate-500">
                    {tt("暂无数据", "No data available")}
                  </div>
                )}
              </div>

              {/* 弹窗底部 */}
              <div className="flex justify-end gap-2 p-4 border-t border-slate-200">
                <Button variant="outline" onClick={() => {
                  setIsPreviewModalOpen(false);
                  setShowAllData(false);
                }}>
                  {tt("关闭", "Close")}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Add Bench Modal */}
      <AnimatePresence>
        {isAddModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center"
          >
            <div className="absolute inset-0 bg-black/20" onClick={() => setIsAddModalOpen(false)} />
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="relative bg-white rounded-2xl shadow-2xl p-6 w-full max-w-lg mx-4"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-slate-900">{tt("新增 Benchmark", "Add New Benchmark")}</h2>
                <button
                  className="p-2 rounded-lg hover:bg-slate-100 text-slate-500"
                  onClick={() => setIsAddModalOpen(false)}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-4">
                {/* 上传模式切换 */}
                <div className="flex gap-2 p-1 bg-slate-100 rounded-lg">
                  <button
                    type="button"
                    onClick={() => setUploadMode("url")}
                    className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                      uploadMode === "url"
                        ? "bg-white text-slate-900 shadow-sm"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    {tt("HuggingFace 链接", "HuggingFace URL")}
                  </button>
                  <button
                    type="button"
                    onClick={() => setUploadMode("file")}
                    className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                      uploadMode === "file"
                        ? "bg-white text-slate-900 shadow-sm"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    {tt("本地上传", "Local Upload")}
                  </button>
                </div>

                <div className="space-y-2">
                  <Label>{tt("Benchmark 名称 *", "Benchmark Name *")}</Label>
                  <Input
                    value={addForm.bench_name}
                    onChange={(e) => setAddForm({ ...addForm, bench_name: e.target.value })}
                    placeholder={uploadMode === "url" ? tt("例如：org/dataset_name", "e.g., org/dataset_name") : tt("例如：my_benchmark", "e.g., my_benchmark")}
                    className="border-slate-200"
                  />
                  {uploadMode === "url" && (
                    <p className="text-xs text-slate-500">{tt("请使用 HuggingFace 格式：org/dataset_name", "Use HuggingFace format: org/dataset_name")}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label>{tt("类型 *", "Type *")}</Label>
                  <select
                    value={addForm.type}
                    onChange={(e) => setAddForm({ ...addForm, type: e.target.value })}
                    className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  >
                    {BENCH_TYPES.map((t) => (
                      <option key={t} value={t}>{benchTypeLabel(t)}</option>
                    ))}
                  </select>
                </div>

                {/* 文件上传模式下显示评测类型选择器 */}
                {uploadMode === "file" && (
                  <div className="space-y-2">
                    <Label>{tt("评测类型 *", "Eval Type *")}</Label>
                    <select
                      value={addForm.eval_type}
                      onChange={(e) => setAddForm({ ...addForm, eval_type: e.target.value })}
                      className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                    >
                      {EVAL_TYPES.map((t) => (
                        <option key={t.value} value={t.value}>{evalTypeLabel(t.value, lang)}</option>
                      ))}
                    </select>
                    <p className="text-xs text-slate-500">{tt("选择数据集的评测格式类型", "Select the evaluation format type of your dataset")}</p>
                  </div>
                )}

                <div className="space-y-2">
                  <Label>{tt("描述 *", "Description *")}</Label>
                  <textarea
                    value={addForm.description}
                    onChange={(e) => setAddForm({ ...addForm, description: e.target.value })}
                    placeholder={tt("描述这个 benchmark 评测什么能力...", "Describe what this benchmark evaluates...")}
                    className="w-full min-h-[80px] rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  />
                </div>

                {uploadMode === "url" ? (
                  <div className="space-y-2">
                    <Label>{tt("数据集链接（可选）", "Dataset URL (optional)")}</Label>
                    <Input
                      value={addForm.dataset_url}
                      onChange={(e) => setAddForm({ ...addForm, dataset_url: e.target.value })}
                      placeholder="https://huggingface.co/datasets/..."
                      className="border-slate-200"
                    />
                    <p className="text-xs text-slate-500">{tt("留空将根据 bench 名自动生成", "Leave empty to auto-generate from bench name")}</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Label>{tt("数据集文件 *", "Dataset File *")}</Label>
                    <div
                      className="border-2 border-dashed border-slate-200 rounded-lg p-6 text-center hover:border-slate-300 transition-colors cursor-pointer"
                      onDragOver={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        e.currentTarget.classList.add("border-blue-400", "bg-blue-50");
                      }}
                      onDragLeave={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        e.currentTarget.classList.remove("border-blue-400", "bg-blue-50");
                      }}
                      onDrop={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        e.currentTarget.classList.remove("border-blue-400", "bg-blue-50");
                        const file = e.dataTransfer.files?.[0];
                        if (file) {
                          setUploadFile(file);
                          if (!addForm.bench_name) {
                            const baseName = file.name.replace(/\.[^.]+$/, "");
                            setAddForm(prev => ({ ...prev, bench_name: baseName }));
                          }
                        }
                      }}
                      onClick={() => document.getElementById("file-upload")?.click()}
                    >
                      <input
                        type="file"
                        accept=".csv,.jsonl,.json,.xlsx,.xls,.txt"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) {
                            setUploadFile(file);
                            // 自动填充 bench_name
                            if (!addForm.bench_name) {
                              const baseName = file.name.replace(/\.[^.]+$/, "");
                              setAddForm(prev => ({ ...prev, bench_name: baseName }));
                            }
                          }
                        }}
                        className="hidden"
                        id="file-upload"
                      />
                      {uploadFile ? (
                        <div className="text-sm text-slate-700 flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <p className="font-medium truncate">{uploadFile.name}</p>
                            <p className="text-xs text-slate-500">{(uploadFile.size / 1024).toFixed(1)} KB</p>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setUploadFile(null);
                            }}
                            className="shrink-0 p-2 rounded-lg hover:bg-red-100 text-red-500 transition-colors"
                            title={tt("删除文件", "Remove file")}
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ) : (
                        <div className="text-sm text-slate-500">
                          <p>{tt("点击选择文件或拖拽到此处", "Click to select or drag file here")}</p>
                          <p className="text-xs mt-1">{tt("支持 .csv, .jsonl, .json, .xlsx, .xls, .txt", "Supports .csv, .jsonl, .json, .xlsx, .xls, .txt")}</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-6 flex gap-3">
                <Button
                  className="flex-1 bg-gradient-to-r from-blue-600 to-violet-600 text-white hover:from-blue-500 hover:to-violet-500"
                  onClick={handleAddBench}
                  disabled={isSubmitting || !addForm.bench_name.trim() || !addForm.description.trim() || (uploadMode === "file" && !uploadFile)}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      {tt("新增中...", "Adding...")}
                    </>
                  ) : (
                    <>
                      <Plus className="w-4 h-4 mr-2" />
                      {tt("新增 Benchmark", "Add Benchmark")}
                    </>
                  )}
                </Button>
                <Button variant="outline" className="border-slate-200" onClick={() => {
                  setIsAddModalOpen(false);
                  setUploadFile(null);
                  setAddForm({ bench_name: "", type: "knowledge", description: "", dataset_url: "", eval_type: "key2_qa" });
                  setUploadMode("url");
                }}>
                  {tt("取消", "Cancel")}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Append Attachment Modal */}
      <AnimatePresence>
        {isAppendModalOpen && activeBench && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center"
          >
            <div className="absolute inset-0 bg-black/20" onClick={() => setIsAppendModalOpen(false)} />
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="relative bg-white rounded-2xl shadow-2xl p-6 w-full max-w-md mx-4"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-slate-900">{tt("追加附件", "Append Attachment")}</h2>
                <button
                  className="p-2 rounded-lg hover:bg-slate-100 text-slate-500"
                  onClick={() => {
                    setIsAppendModalOpen(false);
                    setAppendFile(null);
                  }}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-4">
                <div className="p-3 bg-slate-50 rounded-lg">
                  <div className="text-sm text-slate-600">{tt("目标 Benchmark", "Target Benchmark")}</div>
                  <div className="font-medium text-slate-900">{activeBench.name}</div>
                </div>

                <div className="space-y-2">
                  <Label>{tt("选择附件文件", "Select Attachment File")}</Label>
                  <div
                    className="border-2 border-dashed border-slate-200 rounded-lg p-6 text-center hover:border-slate-300 transition-colors cursor-pointer"
                    onDragOver={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      e.currentTarget.classList.add("border-blue-400", "bg-blue-50");
                    }}
                    onDragLeave={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      e.currentTarget.classList.remove("border-blue-400", "bg-blue-50");
                    }}
                    onDrop={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      e.currentTarget.classList.remove("border-blue-400", "bg-blue-50");
                      const file = e.dataTransfer.files?.[0];
                      if (file) {
                        setAppendFile(file);
                      }
                    }}
                    onClick={() => document.getElementById("append-file-upload")?.click()}
                  >
                    <input
                      type="file"
                      accept=".csv,.jsonl,.json,.xlsx,.xls,.txt"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          setAppendFile(file);
                        }
                      }}
                      className="hidden"
                      id="append-file-upload"
                    />
                    {appendFile ? (
                      <div className="text-sm text-slate-700 flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <p className="font-medium truncate">{appendFile.name}</p>
                          <p className="text-xs text-slate-500">{(appendFile.size / 1024).toFixed(1)} KB</p>
                        </div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setAppendFile(null);
                          }}
                          className="shrink-0 p-2 rounded-lg hover:bg-red-100 text-red-500 transition-colors"
                          title={tt("删除文件", "Remove file")}
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <div className="text-sm text-slate-500">
                        <p>{tt("点击选择文件或拖拽到此处", "Click to select or drag file here")}</p>
                        <p className="text-xs mt-1">{tt("支持 .csv, .jsonl, .json, .xlsx, .xls, .txt", "Supports .csv, .jsonl, .json, .xlsx, .xls, .txt")}</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="mt-6 flex gap-3">
                <Button
                  className="flex-1 bg-gradient-to-r from-blue-600 to-violet-600 text-white hover:from-blue-500 hover:to-violet-500"
                  onClick={handleAppendAttachment}
                  disabled={isAppending || !appendFile}
                >
                  {isAppending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      {tt("追加中...", "Appending...")}
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      {tt("追加附件", "Append Attachment")}
                    </>
                  )}
                </Button>
                <Button variant="outline" className="border-slate-200" onClick={() => {
                  setIsAppendModalOpen(false);
                  setAppendFile(null);
                }}>
                  {tt("取消", "Cancel")}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
