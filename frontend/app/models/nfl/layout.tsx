import { ModelSubnav } from "@/app/components/ModelSubnav";

export default function NflLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <><ModelSubnav league="nfl" />{children}</>;
}
