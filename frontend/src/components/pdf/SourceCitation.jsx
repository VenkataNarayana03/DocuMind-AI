export default function SourceCitation({ source }) {
  return (
    <div className="rounded-md border border-line bg-white p-3 text-sm">
      <div className="mb-1 flex items-center justify-between gap-3 font-semibold">
        <span className="truncate">{source.filename}</span>
        <span className="shrink-0 text-brand">p. {source.page}</span>
      </div>
      <p className="line-clamp-3 leading-5 text-slate-700">{source.text}</p>
    </div>
  );
}
