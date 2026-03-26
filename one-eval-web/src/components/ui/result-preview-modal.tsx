import { useState, useMemo } from "react";
import { SimpleMarkdown } from "./simple-markdown";
import type { Lang } from "@/lib/i18n";

interface PreviewData {
    total: number;
    columns: string[];
    rows: Record<string, any>[];
}

interface ResultPreviewModalProps {
    open: boolean;
    onClose: () => void;
    previewData: PreviewData | null;
    loading: boolean;
    error?: boolean;
    lang: Lang;
    title?: string;
    subtitle?: string;
    onDownload?: (format: "jsonl" | "csv" | "xlsx") => void;
    downloading?: boolean;
    onFetchAll?: () => Promise<PreviewData>;
    fetchingAll?: boolean;
}

const tt = (lang: Lang, zh: string, en: string) => (lang === "zh" ? zh : en);

/** Parse <think...>...</think...>\n<answer...>...</answer...> from generated_ans */
function parseGeneratedAns(text: string): { thinking: string; answer: string } | null {
    if (!text || typeof text !== "string") return null;
    const thinkMatch = text.match(/<think[^>]*>([\s\S]*?)<\/think>\s*/i);
    const answerMatch = text.match(/<answer[^>]*>([\s\S]*?)<\/answer>/i);
    if (thinkMatch && answerMatch) {
        return { thinking: thinkMatch[1].trim(), answer: answerMatch[1].trim() };
    }
    return null;
}

/** Cell renderer with markdown/raw toggle and think/answer split */
function CellContent({
    value,
    lang,
    showRaw,
}: {
    value: unknown;
    lang: Lang;
    showRaw: boolean;
}) {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    const parsed = useMemo(() => parseGeneratedAns(text), [text]);

    if (!parsed) {
        // No think/answer structure — render as-is
        return showRaw ? (
            <pre className="whitespace-pre-wrap break-all text-[11px] font-mono">{text}</pre>
        ) : (
            <SimpleMarkdown content={text} />
        );
    }

    // Has think/answer structure
    return (
        <div className="space-y-2">
            <details className="group">
                <summary className="text-[10px] text-slate-400 cursor-pointer select-none hover:text-slate-600 flex items-center gap-1">
                    <span className="transition-transform group-open:rotate-90 inline-block">&#9654;</span>
                    {tt(lang, "思考过程", "Thinking")}
                </summary>
                <div className="mt-1 pl-3 border-l-2 border-amber-200 bg-amber-50/50 rounded-r p-2">
                    {showRaw ? (
                        <pre className="whitespace-pre-wrap break-all text-[10px] font-mono text-slate-500">{parsed.thinking}</pre>
                    ) : (
                        <div className="max-h-48 overflow-y-auto text-[11px] text-slate-500 scrollbar-thin"><SimpleMarkdown content={parsed.thinking} /></div>
                    )}
                </div>
            </details>
            <div>
                <div className="text-[10px] text-slate-400 mb-1 font-medium">{tt(lang, "输出正文", "Answer")}</div>
                {showRaw ? (
                    <pre className="whitespace-pre-wrap break-all text-[11px] font-mono">{parsed.answer}</pre>
                ) : (
                    <SimpleMarkdown content={parsed.answer} />
                )}
            </div>
        </div>
    );
}

export const ResultPreviewModal = ({
    open,
    onClose,
    previewData,
    loading,
    error = false,
    lang,
    title,
    subtitle,
    onDownload,
    downloading = false,
    onFetchAll,
    fetchingAll = false,
}: ResultPreviewModalProps) => {
    const [showAll, setShowAll] = useState(false);
    const [showRaw, setShowRaw] = useState(false);
    const [allData, setAllData] = useState<PreviewData | null>(null);

    const activeData = showAll && allData ? allData : previewData;

    const handleToggleShowAll = async () => {
        if (showAll) {
            // 切回默认
            setShowAll(false);
            return;
        }
        if (allData) {
            // 已加载过
            setShowAll(true);
            return;
        }
        if (onFetchAll) {
            try {
                const data = await onFetchAll();
                setAllData(data);
                setShowAll(true);
            } catch {}
        }
    };

    if (!open) return null;

    const displayRows = activeData?.rows || [];
    const displayTotal = activeData?.total || 0;
    const displayColumns = activeData?.columns || [];
    const hasMore = activeData ? activeData.rows.length < activeData.total : false;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[85vh] flex flex-col relative z-10 overflow-hidden">
                {/* Header */}
                <div className="p-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50 shrink-0">
                    <div>
                        <h3 className="text-lg font-bold text-slate-900">
                            {title || tt(lang, "评测结果预览", "Result Preview")}
                        </h3>
                        {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
                    </div>
                    <button
                        type="button"
                        onClick={onClose}
                        className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-slate-200 text-slate-400 hover:text-slate-600 text-xl cursor-pointer"
                    >
                        &times;
                    </button>
                </div>

                {/* Toolbar */}
                {!loading && previewData && (
                    <div className="px-5 py-2 border-b border-slate-100 bg-slate-50/30 flex items-center gap-3 shrink-0">
                        <span className="text-xs text-slate-400 mr-auto">
                            {tt(
                                lang,
                                `共 ${displayTotal} 条记录，${showAll ? "显示全部" : `显示前 ${displayRows.length} 条`}`,
                                `Total ${displayTotal} records, ${showAll ? "showing all" : `showing ${displayRows.length}`}`
                            )}
                        </span>
                        {onFetchAll && displayTotal > 5 && (
                            <button
                                type="button"
                                onClick={handleToggleShowAll}
                                disabled={fetchingAll}
                                className="text-[11px] px-2.5 py-1 rounded-md border transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                                style={{
                                    borderColor: showAll ? "#bfdbfe" : undefined,
                                    backgroundColor: showAll ? "#eff6ff" : "white",
                                    color: showAll ? "#2563eb" : undefined,
                                }}
                            >
                                {fetchingAll
                                    ? tt(lang, "加载中...", "Loading...")
                                    : showAll
                                        ? tt(lang, "显示前 5 条", "Show first 5")
                                        : tt(lang, `显示全部 (${displayTotal})`, `Show all (${displayTotal})`)}
                            </button>
                        )}
                        <button
                            type="button"
                            onClick={() => setShowRaw(!showRaw)}
                            className="text-[11px] px-2.5 py-1 rounded-md border transition-colors cursor-pointer"
                            style={{
                                borderColor: showRaw ? "#bfdbfe" : undefined,
                                backgroundColor: showRaw ? "#eff6ff" : "white",
                                color: showRaw ? "#2563eb" : undefined,
                            }}
                        >
                            {showRaw ? tt(lang, "Markdown 渲染", "Rendered") : tt(lang, "原始内容", "Raw")}
                        </button>
                    </div>
                )}

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-5">
                    {loading && (
                        <div className="text-center py-10 text-sm text-slate-400">
                            {tt(lang, "加载中...", "Loading...")}
                        </div>
                    )}
                    {!loading && error && !previewData && (
                        <div className="text-center py-10 text-sm text-red-400">
                            {tt(lang, "加载失败，请重试", "Failed to load, please retry")}
                        </div>
                    )}
                    {!loading && previewData && (
                        <div className="overflow-x-auto">
                            <table className="w-full text-xs border-collapse">
                                <thead>
                                    <tr className="border-b border-slate-200">
                                        <th className="px-2 py-2 text-left font-medium text-slate-400 w-10 sticky top-0 bg-white">#</th>
                                        {displayColumns.map((col: string) => (
                                            <th
                                                key={col}
                                                className="px-2 py-2 text-left font-medium text-slate-600 whitespace-nowrap sticky top-0 bg-white"
                                            >
                                                {col}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {displayRows.map((row: any, ri: number) => (
                                        <tr key={ri} className="border-b border-slate-100 hover:bg-slate-50">
                                            <td className="px-2 py-1.5 text-slate-400 font-mono sticky left-0 bg-inherit">{ri + 1}</td>
                                            {displayColumns.map((col: string) => (
                                                <td key={col} className="px-2 py-2 text-slate-700 align-top min-w-[200px] max-w-[600px]">
                                                    <CellContent value={row[col]} lang={lang} showRaw={showRaw} />
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>

                {/* Footer */}
                {onDownload && (
                    <div className="p-4 border-t border-slate-100 bg-slate-50/50 flex items-center justify-between">
                        <span className="text-xs text-slate-400">
                            {tt(lang, "选择格式下载完整数据", "Download full data in chosen format")}
                        </span>
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                onClick={() => onDownload("jsonl")}
                                disabled={downloading}
                                className="text-[11px] px-3 py-1.5 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                JSONL
                            </button>
                            <button
                                type="button"
                                onClick={() => onDownload("csv")}
                                disabled={downloading}
                                className="text-[11px] px-3 py-1.5 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                CSV
                            </button>
                            <button
                                type="button"
                                onClick={() => onDownload("xlsx")}
                                disabled={downloading}
                                className="text-[11px] px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                XLSX
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
