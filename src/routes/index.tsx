import { createFileRoute } from "@tanstack/react-router";
import { ClientOnly } from "@tanstack/react-router";
import { Suspense, lazy } from "react";
import { ScanLine } from "lucide-react";

const CameraCounter = lazy(() => import("@/components/CameraCounter"));

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "FluxoCam — Contagem de pessoas por câmera" },
      {
        name: "description",
        content:
          "Conecte câmeras, detecte pessoas em tempo real no navegador, capture fotos automáticas e conte quem cruza a linha X → Y.",
      },
      { property: "og:title", content: "FluxoCam — Contagem de pessoas por câmera" },
      {
        property: "og:description",
        content:
          "Análise de vídeo em tempo real: detecção de pessoas, foto automática e contagem por cruzamento de linha.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Skeleton() {
  return (
    <div className="aspect-video w-full animate-pulse rounded-2xl border border-border bg-surface" />
  );
}

function Index() {
  return (
    <main className="mx-auto min-h-screen w-full max-w-7xl px-4 py-8 sm:px-6 lg:py-12">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2 text-accent">
            <ScanLine className="h-5 w-5" />
            <span className="font-mono text-xs uppercase tracking-[0.3em]">FluxoCam</span>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            Contagem de pessoas por câmera
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            Arraste os pontos <strong className="text-foreground">X</strong> e{" "}
            <strong className="text-foreground">Y</strong> sobre o vídeo para definir a linha de
            passagem. Cada pessoa que cruzar é fotografada e contabilizada — tudo processado no seu
            navegador.
          </p>
        </div>
      </header>

      <ClientOnly fallback={<Skeleton />}>
        <Suspense fallback={<Skeleton />}>
          <CameraCounter />
        </Suspense>
      </ClientOnly>
    </main>
  );
}
