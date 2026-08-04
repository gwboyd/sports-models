export default function ModelsLoading() {
  return (
    <div aria-live="polite" className="mx-auto flex w-full max-w-7xl flex-col gap-4 px-4 py-8 opacity-60 sm:px-6 lg:px-8">
      <div className="h-8 w-48 animate-pulse rounded-lg bg-slate-200" />
      <div className="h-32 animate-pulse rounded-lg bg-slate-200" />
      <span className="sr-only">Loading model data…</span>
    </div>
  );
}
