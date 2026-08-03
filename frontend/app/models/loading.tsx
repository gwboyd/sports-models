export default function ModelsLoading() {
  return (
    <div aria-live="polite" className="overflow-y-auto flex flex-col gap-4 p-6 pb-28 lg:pb-24 lg:px-12 opacity-60">
      <div className="h-8 w-48 rounded bg-gray-800" />
      <div className="h-36 rounded bg-gray-800" />
      <span className="sr-only">Loading model data…</span>
    </div>
  );
}
