import { ModelTabs } from "@/app/components/ModelTabs";

export default function ModelsLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="min-h-dvh">
      <header className="sticky top-0 z-40 border-b border-[var(--border)] bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-2 sm:px-6 lg:px-8">
          <a href="/models/nfl" className="whitespace-nowrap text-base font-bold tracking-tight text-[var(--ink)] sm:text-lg">
            Boyd&apos;s Picks
          </a>
          <ModelTabs />
        </div>
      </header>
      {children}
    </div>
  );
}
