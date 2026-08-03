import { ModelTabs } from "@/app/components/ModelTabs";

export default function ModelsLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="flex flex-col overflow-hidden">
      <ModelTabs />
      {children}
    </div>
  );
}
