import { ModelSubnav } from "@/app/components/ModelSubnav";

export default function CfbLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <><ModelSubnav league="cfb" />{children}</>;
}
