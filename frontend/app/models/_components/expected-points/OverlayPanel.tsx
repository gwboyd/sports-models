"use client";

import { useEffect, useRef, type ReactNode } from "react";

export function OverlayPanel({ title, description, onClose, children }: {
  title: string;
  description?: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCloseRef.current();
    };
    const syncVisualViewport = () => {
      const overlay = overlayRef.current;
      if (!overlay) return;
      const viewport = window.visualViewport;
      overlay.style.top = `${viewport?.offsetTop ?? 0}px`;
      overlay.style.height = `${viewport?.height ?? window.innerHeight}px`;
    };

    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", syncVisualViewport);
    window.visualViewport?.addEventListener("resize", syncVisualViewport);
    window.visualViewport?.addEventListener("scroll", syncVisualViewport);

    const body = document.body;
    const root = document.documentElement;
    const scrollY = window.scrollY;
    const previousBodyStyles = {
      overflow: body.style.overflow,
      position: body.style.position,
      top: body.style.top,
      width: body.style.width,
    };
    const previousRootOverflow = root.style.overflow;

    root.style.overflow = "hidden";
    body.style.overflow = "hidden";
    body.style.position = "fixed";
    body.style.top = `-${scrollY}px`;
    body.style.width = "100%";
    syncVisualViewport();

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", syncVisualViewport);
      window.visualViewport?.removeEventListener("resize", syncVisualViewport);
      window.visualViewport?.removeEventListener("scroll", syncVisualViewport);
      root.style.overflow = previousRootOverflow;
      body.style.overflow = previousBodyStyles.overflow;
      body.style.position = previousBodyStyles.position;
      body.style.top = previousBodyStyles.top;
      body.style.width = previousBodyStyles.width;
      window.scrollTo(0, scrollY);
    };
  }, []);

  return (
    <div ref={overlayRef} className="fixed inset-x-0 top-0 z-50 flex h-dvh items-end justify-center overflow-hidden bg-slate-950/35 p-0 sm:items-center sm:p-6" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section role="dialog" aria-modal="true" aria-labelledby="overlay-title" className="flex h-[88%] w-full flex-col overflow-hidden rounded-t-lg bg-white shadow-xl sm:h-auto sm:max-h-[88%] sm:max-w-xl sm:rounded-lg">
        <header className="flex shrink-0 items-start justify-between gap-4 border-b border-[var(--border)] px-4 py-3.5">
          <div>
            <h2 id="overlay-title" className="text-lg font-semibold text-[var(--ink)]">{title}</h2>
            {description ? <p className="mt-1 text-sm text-[var(--muted)]">{description}</p> : null}
          </div>
          <button type="button" aria-label="Close" onClick={onClose} className="flex min-h-11 min-w-11 items-center justify-center rounded-lg text-2xl text-slate-500 hover:bg-slate-100">×</button>
        </header>
        <div className="min-h-0 flex-1 overflow-hidden p-4">{children}</div>
      </section>
    </div>
  );
}
