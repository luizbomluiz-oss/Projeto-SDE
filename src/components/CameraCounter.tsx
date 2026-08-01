import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import Hls from "hls.js";
import { toast } from "sonner";
import {
  Ban,
  Camera,
  Download,
  FolderOpen,
  ImagePlus,
  Link2,
  Loader2,
  LogIn,
  LogOut,
  Play,
  RotateCcw,
  Square,
  Trash2,
  Users,
  Video,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  PeopleTracker,
  type Box,
  type PassEvent,
  type Point,
  type Zone,
} from "@/lib/people-counter";
import { detectObjectInFrame } from "@/lib/object-filter.functions";

const CSS_VAR_ACCENT = "--counter-accent";

type ZoneKey = "entry" | "exit";
type DragMode = { key: ZoneKey; type: "move" | "resize"; dx: number; dy: number } | null;

function readToken(name: string, fallback: string) {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function clamp01(v: number) {
  return Math.min(1, Math.max(0, v));
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => resolve("");
    reader.readAsDataURL(blob);
  });
}

export default function CameraCounter() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const snapRef = useRef<HTMLCanvasElement | null>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const trackerRef = useRef(new PeopleTracker());
  const trackedPeopleRef = trackerRef;
  const rafRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const hlsRef = useRef<Hls | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const thresholdRef = useRef(0.45);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const dirRef = useRef<any>(null);
  const dragRef = useRef<DragMode>(null);

  const photoQueueRef = useRef<
    {
      id: number;
      x: number;
      y: number;
      box: Box;
      countWeight: number;
      canvasFrame?: HTMLVideoElement | HTMLCanvasElement;
    }[]
  >([]);
  const isProcessingQueueRef = useRef(false);

  const zonesInit = useMemo(
    () => ({
      entry: { x: 0.06, y: 0.2, w: 0.3, h: 0.6 } as Zone,
      exit: { x: 0.64, y: 0.2, w: 0.3, h: 0.6 } as Zone,
    }),
    [],
  );
  const zonesRef = useRef(zonesInit);

  const [modelState, setModelState] = useState<"idle" | "loading" | "ready">("idle");
  const [running, setRunning] = useState(false);
  const [hasSource, setHasSource] = useState(false);
  const [threshold, setThreshold] = useState(45);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [streamUrl, setStreamUrl] = useState("");
  const [live, setLive] = useState(0);
  const [fps, setFps] = useState(0);
  const [count, setCount] = useState(0);
  const [ignored, setIgnored] = useState(0);
  const [events, setEvents] = useState<PassEvent[]>([]);
  const [zones, setZones] = useState(zonesInit);
  const [saidaPolygonPoints, setSaidaPolygonPoints] = useState<Point[]>([]);
  const [folderName, setFolderName] = useState<string>("");
  const [filterEnabled, setFilterEnabled] = useState(false);
  const [refImage, setRefImage] = useState<string>("");
  const [objectHint, setObjectHint] = useState("");

  const filterRef = useRef(false);
  const refImageRef = useRef("");
  const hintRef = useRef("");
  const eventsRef = useRef<PassEvent[]>([]);
  const polygonRef = useRef<Point[]>([]);
  const dragVertexIndexRef = useRef<number | null>(null);
  /** photos still pending export (only used when no local folder is available) */
  const blobsRef = useRef<Map<string, Blob>>(new Map());
  const frameRef = useRef(0);

  useEffect(() => {
    polygonRef.current = saidaPolygonPoints;
  }, [saidaPolygonPoints]);

  useEffect(() => {
    eventsRef.current = events;
  }, [events]);

  useEffect(() => {
    thresholdRef.current = threshold / 100;
  }, [threshold]);
  useEffect(() => {
    zonesRef.current = zones;
  }, [zones]);
  useEffect(() => {
    filterRef.current = filterEnabled;
  }, [filterEnabled]);
  useEffect(() => {
    refImageRef.current = refImage;
  }, [refImage]);
  useEffect(() => {
    hintRef.current = objectHint;
  }, [objectHint]);

  const supportsFolder =
    typeof window !== "undefined" && "showDirectoryPicker" in (window as object);

  const stopSource = useCallback(() => {
    hlsRef.current?.destroy();
    hlsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.removeAttribute("src");
      videoRef.current.load();
    }
  }, []);

  useEffect(() => {
    return () => {
      runningRef.current = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      stopSource();
    };
  }, [stopSource]);

  const chooseFolder = useCallback(async () => {
    if (!supportsFolder) {
      toast.error(
        "Seu navegador não permite escolher pasta. Use o botão de exportar fotos (.zip).",
      );
      return;
    }
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const handle = await (window as any).showDirectoryPicker({ mode: "readwrite" });
      const perm = await handle.requestPermission?.({ mode: "readwrite" });
      if (perm && perm !== "granted") {
        toast.error("Permissão de escrita negada");
        return;
      }
      dirRef.current = handle;
      setFolderName(handle.name);
      toast.success(`Fotos serão salvas em "${handle.name}"`);
    } catch (err) {
      const e = err as DOMException;
      if (e?.name === "AbortError") return;
      toast.error(
        "Não foi possível abrir a pasta aqui (bloqueado no preview embutido). Abra o site em uma aba própria ou use a exportação .zip.",
      );
    }
  }, [supportsFolder]);

  const exportZip = useCallback(async () => {
    const entries = [...blobsRef.current.entries()];
    if (!entries.length) {
      toast.error("Nenhuma foto para exportar ainda");
      return;
    }
    const JSZip = (await import("jszip")).default;
    const zip = new JSZip();
    for (const [name, blob] of entries) zip.file(name, blob);
    const out = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(out);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fluxocam-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.zip`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`${entries.length} foto(s) exportada(s)`);
  }, []);

  /** Writes the blob to disk (if folder chosen) and keeps it in memory for .zip export & UI preview. */
  const savePhoto = useCallback(async (blob: Blob | null, fileName: string) => {
    if (!blob) return "";
    blobsRef.current.set(fileName, blob);
    const dir = dirRef.current;
    if (!dir) return "";
    try {
      const file = await dir.getFileHandle(fileName, { create: true });
      const writable = await file.createWritable();
      await writable.write(blob);
      await writable.close();
      return fileName;
    } catch (err) {
      console.error(err);
      return "";
    }
  }, []);

  /** Drops the preview URL of an event so the V8 GC can reclaim the blob. */
  const releasePreview = useCallback((id: string, url: string) => {
    if (url) URL.revokeObjectURL(url);
    setEvents((prev) => prev.map((e) => (e.id === id ? { ...e, image: "" } : e)));
  }, []);

  const ensureModel = useCallback(async () => {
    if (modelRef.current) return modelRef.current;
    setModelState("loading");
    await tf.ready();
    const model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
    modelRef.current = model;
    setModelState("ready");
    return model;
  }, []);

  const startWebcam = useCallback(
    async (id?: string) => {
      try {
        stopSource();
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            ...(id ? { deviceId: { exact: id } } : { facingMode: "environment" }),
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 },
          },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setHasSource(true);
        const list = (await navigator.mediaDevices.enumerateDevices()).filter(
          (d) => d.kind === "videoinput",
        );
        setDevices(list);
        const active = stream.getVideoTracks()[0]?.getSettings().deviceId;
        if (active) setDeviceId(active);
        toast.success("Câmera conectada");
      } catch (err) {
        console.error(err);
        toast.error("Não foi possível acessar a câmera. Verifique as permissões.");
      }
    },
    [stopSource],
  );

  const startStream = useCallback(async () => {
    const url = streamUrl.trim();
    if (!url) return toast.error("Informe a URL do stream");
    stopSource();
    const video = videoRef.current;
    if (!video) return;
    try {
      if (url.includes(".m3u8") && Hls.isSupported()) {
        const hls = new Hls({ lowLatencyMode: true });
        hlsRef.current = hls;
        hls.loadSource(url);
        hls.attachMedia(video);
        hls.on(Hls.Events.ERROR, (_e, data) => {
          if (data.fatal) toast.error("Falha no stream HLS");
        });
      } else {
        video.crossOrigin = "anonymous";
        video.src = url;
      }
      await video.play().catch(() => undefined);
      setHasSource(true);
      toast.success("Stream conectado");
    } catch (err) {
      console.error(err);
      toast.error("Não foi possível abrir o stream");
    }
  }, [stopSource, streamUrl]);

  /** Async snapshot: never blocks the render loop. */
  const takeSnapshot = useCallback((box: Box): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video) return Promise.resolve(null);
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return Promise.resolve(null);
    try {
      ctx.drawImage(video, 0, 0, w, h);
      ctx.lineWidth = Math.max(2, w * 0.005);
      ctx.strokeStyle = readToken(CSS_VAR_ACCENT, "#7dd3fc");
      ctx.strokeRect(box.x * w, box.y * h, box.w * w, box.h * h);
    } catch {
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      try {
        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.7);
      } catch {
        resolve(null);
      }
    });
  }, []);

  const draw = useCallback((tracks: { cx: number; cy: number; box: Box; id: number }[]) => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const rect = video.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = Math.max(1, Math.round(rect.width));
      canvas.height = Math.max(1, Math.round(rect.height));
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width: W, height: H } = canvas;
    ctx.clearRect(0, 0, W, H);

    const accent = readToken(CSS_VAR_ACCENT, "#7dd3fc");

    // 1. Polygon ROI for Saída Region (using Path2D and isPointInPath)
    const polyPoints =
      polygonRef.current && polygonRef.current.length >= 3
        ? polygonRef.current
        : [
            { x: zonesRef.current.exit.x, y: zonesRef.current.exit.y },
            { x: zonesRef.current.exit.x + zonesRef.current.exit.w, y: zonesRef.current.exit.y },
            {
              x: zonesRef.current.exit.x + zonesRef.current.exit.w,
              y: zonesRef.current.exit.y + zonesRef.current.exit.h,
            },
            { x: zonesRef.current.exit.x, y: zonesRef.current.exit.y + zonesRef.current.exit.h },
          ];

    const pathSaida = new Path2D();
    polyPoints.forEach((pt, idx) => {
      if (idx === 0) pathSaida.moveTo(pt.x * W, pt.y * H);
      else pathSaida.lineTo(pt.x * W, pt.y * H);
    });
    pathSaida.closePath();

    ctx.save();
    ctx.strokeStyle = accent;
    ctx.fillStyle = accent;
    ctx.globalAlpha = 0.15;
    ctx.fill(pathSaida);
    ctx.globalAlpha = 1;
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.stroke(pathSaida);
    ctx.setLineDash([]);

    // Draw vertex handles & labels
    polyPoints.forEach((pt, i) => {
      ctx.beginPath();
      ctx.arc(pt.x * W, pt.y * H, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = accent;
      ctx.stroke();

      ctx.font = "bold 10px ui-sans-serif, system-ui";
      ctx.fillStyle = accent;
      ctx.fillText(`P${i + 1}`, pt.x * W + 8, pt.y * H + 4);
    });

    if (polyPoints.length > 0) {
      ctx.font = "600 12px ui-sans-serif, system-ui";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillStyle = accent;
      const isCustom = polygonRef.current && polygonRef.current.length >= 3;
      ctx.fillText(
        isCustom ? `ÁREA DE SAÍDA (POLÍGONO - ${polyPoints.length} VÉRTICES)` : "SAÍDA (PADRÃO)",
        polyPoints[0].x * W + 6,
        polyPoints[0].y * H + 6,
      );
    }
    ctx.restore();

    // Render active tracks
    ctx.strokeStyle = accent;
    ctx.fillStyle = accent;
    ctx.lineWidth = 2;
    tracks.forEach((t) => {
      ctx.strokeRect(t.box.x * W, t.box.y * H, t.box.w * W, t.box.h * H);
      ctx.beginPath();
      ctx.arc(t.cx * W, t.cy * H, 4, 0, Math.PI * 2);
      ctx.fill();

      // Check if barycenter is inside polygon using Path2D context method
      const isInside = ctx.isPointInPath(pathSaida, t.cx * W, t.cy * H);
      if (isInside) {
        ctx.beginPath();
        ctx.arc(t.cx * W, t.cy * H, 8, 0, Math.PI * 2);
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.font = "600 12px ui-sans-serif, system-ui";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`#${t.id}`, t.box.x * W + 2, t.box.y * H - 4);
    });
  }, []);

  const registerPass = useCallback(
    async (trackId: number, blob: Blob | null, countWeight = 1) => {
      const at = Date.now();
      const id = `${trackId}-${at}-${Math.random().toString(36).slice(2, 7)}`;
      const useFilter = filterRef.current && !!refImageRef.current && !!blob;
      const previewUrl = blob ? URL.createObjectURL(blob) : "";

      setEvents((prev) =>
        [
          {
            id,
            trackId,
            at,
            image: previewUrl,
            status: useFilter ? "checking" : "ok",
          } as PassEvent,
          ...prev,
        ].slice(0, 60),
      );

      /** writes to disk, updates status in UI, keeps preview intact */
      const persist = async (fileName: string) => {
        const saved = await savePhoto(blob, fileName);
        setEvents((prev) =>
          prev.map((e) => (e.id === id ? { ...e, fileName, savedTo: saved || undefined } : e)),
        );
      };

      if (!useFilter) {
        setCount((c) => c + countWeight);
        await persist(`passagem-${trackId}-${at}.jpg`);
        return;
      }

      try {
        const frame = blob ? await blobToDataUrl(blob) : "";
        const result = await detectObjectInFrame({
          data: { reference: refImageRef.current, frame, hint: hintRef.current || undefined },
        });
        if (result.hasObject) {
          setIgnored((c) => c + 1);
          setEvents((prev) =>
            prev.map((e) =>
              e.id === id
                ? { ...e, status: "ignored", reason: result.note || "objeto encontrado" }
                : e,
            ),
          );
          // filtered photos are also stored, flagged in the file name
          await persist(`ignorado-${trackId}-${at}.jpg`);
          return;
        }
        setCount((c) => c + countWeight);
        setEvents((prev) => prev.map((e) => (e.id === id ? { ...e, status: "ok" } : e)));
        await persist(`passagem-${trackId}-${at}.jpg`);
      } catch (err) {
        console.error(err);
        const msg = err instanceof Error ? err.message : "";
        toast.error(
          msg === "RATE_LIMIT"
            ? "Limite de uso da IA atingido, tente novamente em instantes."
            : msg === "NO_CREDITS"
              ? "Créditos de IA esgotados. Adicione créditos no seu workspace."
              : "Falha ao analisar o objeto com a IA.",
        );
        setCount((c) => c + countWeight);
        setEvents((prev) => prev.map((e) => (e.id === id ? { ...e, status: "ok" } : e)));
        await persist(`passagem-${trackId}-${at}.jpg`);
      }
    },
    [savePhoto],
  );

  const processPhotoQueue = useCallback(async () => {
    if (isProcessingQueueRef.current) return;
    isProcessingQueueRef.current = true;

    while (photoQueueRef.current.length > 0) {
      const item = photoQueueRef.current[0];
      if (item) {
        const blob = await takeSnapshot(item.box);
        await registerPass(item.id, blob, item.countWeight);
      }
      photoQueueRef.current.shift();
    }

    isProcessingQueueRef.current = false;
  }, [takeSnapshot, registerPass]);

  const DETECT_EVERY = 2;

  const loop = useCallback(async () => {
    const video = videoRef.current;
    const model = modelRef.current;
    if (!runningRef.current || !video || !model) return;

    frameRef.current += 1;
    const shouldDetect = frameRef.current % DETECT_EVERY === 0;

    if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
      if (shouldDetect) {
        const started = performance.now();
        let raw: cocoSsd.DetectedObject[] = [];
        tf.engine().startScope();
        try {
          raw = await model.detect(video, 20, thresholdRef.current);
        } catch (err) {
          console.error(err);
        } finally {
          tf.engine().endScope();
        }
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const boxes: Box[] = raw
          .filter((d) => d.class === "person" && d.score >= thresholdRef.current)
          .map((d) => ({
            x: d.bbox[0] / vw,
            y: d.bbox[1] / vh,
            w: d.bbox[2] / vw,
            h: d.bbox[3] / vh,
            score: d.score,
          }));

        const { passes, tracks } = trackerRef.current.update(
          boxes,
          zonesRef.current,
          polygonRef.current,
        );
        setLive(tracks.length);
        draw(tracks);

        passes.forEach((p) => {
          const track = tracks.find((t) => t.id === p.trackId);
          if (!track) return;
          photoQueueRef.current.push({
            id: track.id,
            x: track.cx,
            y: track.cy,
            box: track.box,
            countWeight: p.countWeight,
            canvasFrame: video,
          });
        });

        if (passes.length > 0) {
          void processPhotoQueue();
        }

        const elapsed = performance.now() - started;
        setFps(Math.round(1000 / Math.max(elapsed, 1)));
      } else {
        draw(trackerRef.current.getTracks());
      }
    }

    rafRef.current = requestAnimationFrame(() => {
      void loop();
    });
  }, [draw, processPhotoQueue]);

  const start = useCallback(async () => {
    if (!hasSource) return toast.error("Conecte uma câmera ou stream primeiro");
    if (!dirRef.current) {
      toast.info('Sem pasta local: use "Exportar fotos (.zip)" para baixar as capturas.');
    }

    if (filterEnabled && !refImage) {
      toast.error("Envie a imagem do objeto para usar o filtro");
      return;
    }
    await ensureModel();
    trackerRef.current.reset();
    runningRef.current = true;
    setRunning(true);
    void loop();
  }, [ensureModel, filterEnabled, hasSource, loop, refImage]);

  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    setLive(0);
  }, []);

  const resetCounts = useCallback(() => {
    trackerRef.current.reset();
    setCount(0);
    setIgnored(0);
    eventsRef.current.forEach((e) => e.image && URL.revokeObjectURL(e.image));
    blobsRef.current.clear();
    setEvents([]);
    toast.success("Contagem zerada");
  }, []);

  const onRefFile = useCallback((file?: File | null) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setRefImage(String(reader.result || ""));
    reader.readAsDataURL(file);
  }, []);

  const pointFromEvent = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: clamp01((e.clientX - rect.left) / rect.width),
      y: clamp01((e.clientY - rect.top) / rect.height),
    };
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const p = pointFromEvent(e);
    const currentPoly = polygonRef.current;

    // 1. Check if clicking near an existing polygon vertex (radius 0.05 normalized)
    if (currentPoly && currentPoly.length > 0) {
      for (let i = 0; i < currentPoly.length; i++) {
        const dist = Math.hypot(p.x - currentPoly[i].x, p.y - currentPoly[i].y);
        if (dist <= 0.05) {
          dragVertexIndexRef.current = i;
          e.currentTarget.setPointerCapture(e.pointerId);
          return;
        }
      }
    }

    // 2. Click on canvas appends a new vertex point to saidaPolygonPoints
    setSaidaPolygonPoints((prev) => [...prev, p]);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const p = pointFromEvent(e);
    if (dragVertexIndexRef.current !== null) {
      const idx = dragVertexIndexRef.current;
      setSaidaPolygonPoints((prev) => {
        const next = [...prev];
        if (idx >= 0 && idx < next.length) {
          next[idx] = p;
        }
        return next;
      });
      return;
    }
  };

  const onPointerUp = () => {
    dragVertexIndexRef.current = null;
  };

  const clearPolygon = useCallback(() => {
    setSaidaPolygonPoints([]);
    toast.info("Área de saída reiniciada (utilizando retângulo padrão)");
  }, []);

  useEffect(() => {
    if (!running) draw(trackerRef.current.getTracks());
  }, [zones, saidaPolygonPoints, running, draw]);

  const stats = useMemo(
    () => [
      { label: "Pessoas contadas", value: count, icon: Users },
      { label: "Ignoradas pelo filtro", value: ignored, icon: Ban },
      { label: "Pessoas em cena", value: live, icon: Video },
    ],
    [count, ignored, live],
  );

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <section className="space-y-4">
          <div className="relative overflow-hidden rounded-2xl border border-border bg-surface shadow-panel">
            <video
              ref={videoRef}
              playsInline
              muted
              autoPlay
              className="block aspect-video w-full bg-black object-contain"
            />

            <canvas
              ref={overlayRef}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
            />
            <canvas ref={snapRef} className="hidden" />
            {!hasSource && (
              <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 text-center">
                <Camera className="h-8 w-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  Conecte uma câmera ou um stream para começar
                </p>
              </div>
            )}
            <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 rounded-full border border-border bg-background/80 px-3 py-1 text-xs backdrop-blur">
              <span
                className={`h-2 w-2 rounded-full ${running ? "animate-pulse bg-accent" : "bg-muted-foreground"}`}
              />
              {running ? `ao vivo · ${fps} fps` : "parado"}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={running ? stop : start} disabled={modelState === "loading"}>
              {modelState === "loading" ? (
                <Loader2 className="animate-spin" />
              ) : running ? (
                <Square />
              ) : (
                <Play />
              )}
              {modelState === "loading"
                ? "Carregando modelo…"
                : running
                  ? "Parar análise"
                  : "Iniciar análise"}
            </Button>
            <Button variant="outline" onClick={resetCounts}>
              <RotateCcw /> Zerar
            </Button>
            <Button
              variant="outline"
              onClick={clearPolygon}
              title="Limpar vértices da área de saída e voltar ao padrão"
            >
              <Trash2 /> Limpar Área de Saída
            </Button>
            <div className="flex min-w-[200px] flex-1 items-center gap-3">
              <Label className="whitespace-nowrap text-xs text-muted-foreground">
                Confiança {threshold}%
              </Label>
              <Slider
                value={[threshold]}
                min={20}
                max={90}
                step={5}
                onValueChange={(v) => setThreshold(v[0])}
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {stats.map((s) => (
              <div
                key={s.label}
                className="rounded-xl border border-border bg-surface p-4 shadow-panel"
              >
                <s.icon className="mb-2 h-4 w-4 text-accent" />
                <p className="font-mono text-3xl font-semibold tabular-nums">{s.value}</p>
                <p className="mt-1 text-xs text-muted-foreground">{s.label}</p>
              </div>
            ))}
          </div>
        </section>

        <aside className="space-y-4">
          <div className="rounded-2xl border border-border bg-surface p-4 shadow-panel">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-muted-foreground">
              Fotos
            </h2>
            <Button className="w-full" variant="secondary" onClick={() => void chooseFolder()}>
              <FolderOpen /> {folderName ? `Pasta: ${folderName}` : "Escolher pasta local"}
            </Button>
            <Button className="mt-2 w-full" onClick={() => void exportZip()}>
              <Download /> Exportar fotos (.zip)
            </Button>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              {supportsFolder
                ? "Opcional: escolha uma pasta para gravar direto no disco (pode ser bloqueado dentro do preview — abra o site em uma aba própria). Se não funcionar, exporte tudo em .zip."
                : "Este navegador não permite gravar em pasta local. As capturas ficam na sessão e podem ser baixadas em .zip."}
            </p>
          </div>

          <div className="rounded-2xl border border-border bg-surface p-4 shadow-panel">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-muted-foreground">
              Áreas de entrada e saída
            </h2>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Arraste os retângulos no vídeo para posicionar. Use o canto inferior direito de cada
              área para redimensionar. A foto é tirada quando alguém passa pela área de{" "}
              <span className="text-foreground">entrada</span> e chega na de{" "}
              <span className="text-foreground">saída</span>.
            </p>
            <div className="mt-3 space-y-3">
              {[
                { key: "entry" as ZoneKey, label: "Entrada", Icon: LogIn },
                { key: "exit" as ZoneKey, label: "Saída", Icon: LogOut },
              ].map(({ key, label, Icon }) => (
                <div key={key} className="rounded-lg border border-border p-2">
                  <div className="mb-2 flex items-center gap-2 text-xs font-medium">
                    <Icon className="h-3.5 w-3.5" /> {label}
                  </div>
                  <div className="grid grid-cols-4 gap-1.5">
                    {[
                      { f: "x" as const, l: "X" },
                      { f: "y" as const, l: "Y" },
                      { f: "w" as const, l: "L" },
                      { f: "h" as const, l: "A" },
                    ].map(({ f, l }) => (
                      <label key={f} className="space-y-1">
                        <span className="block text-[10px] uppercase text-muted-foreground">
                          {l} %
                        </span>
                        <Input
                          type="number"
                          min={f === "w" || f === "h" ? 5 : 0}
                          max={100}
                          className="h-8 px-1.5 text-center font-mono text-xs tabular-nums"
                          value={Math.round(zones[key][f] * 100)}
                          onChange={(e) => {
                            const v = clamp01((Number(e.target.value) || 0) / 100);
                            setZones((prev) => {
                              const z = { ...prev[key], [f]: v };
                              if (f === "w" || f === "h") {
                                z[f] = Math.max(0.05, Math.min(z[f], 1));
                              }
                              z.x = Math.min(z.x, 1 - z.w);
                              z.y = Math.min(z.y, 1 - z.h);
                              return { ...prev, [key]: z };
                            });
                          }}
                        />
                      </label>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <Button className="mt-3 w-full" variant="ghost" onClick={() => setZones(zonesInit)}>
              <RotateCcw /> Restaurar áreas
            </Button>
          </div>

          <div className="rounded-2xl border border-border bg-surface p-4 shadow-panel">
            <div className="mb-3 flex items-center justify-between gap-2">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
                Filtrar por objeto
              </h2>
              <Switch checked={filterEnabled} onCheckedChange={setFilterEnabled} />
            </div>
            <p className="mb-3 text-xs leading-relaxed text-muted-foreground">
              Envie a foto de um objeto. A IA analisa cada passagem e não conta a pessoa que estiver
              com esse objeto.
            </p>
            <div className="space-y-3">
              <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-border px-3 py-4 text-xs text-muted-foreground hover:border-accent">
                <ImagePlus className="h-4 w-4" />
                {refImage ? "Trocar imagem do objeto" : "Enviar imagem do objeto"}
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => onRefFile(e.target.files?.[0])}
                />
              </label>
              {refImage && (
                <img
                  src={refImage}
                  alt="Objeto de referência"
                  className="h-24 w-full rounded-lg object-contain"
                />
              )}
              <Input
                placeholder="Descrição opcional (ex: capacete amarelo)"
                value={objectHint}
                onChange={(e) => setObjectHint(e.target.value)}
              />
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-surface p-4 shadow-panel">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-muted-foreground">
              Fonte de vídeo
            </h2>
            <Tabs defaultValue="webcam">
              <TabsList className="w-full">
                <TabsTrigger value="webcam" className="flex-1">
                  <Camera className="mr-1 h-3.5 w-3.5" /> Câmera
                </TabsTrigger>
                <TabsTrigger value="stream" className="flex-1">
                  <Link2 className="mr-1 h-3.5 w-3.5" /> Stream
                </TabsTrigger>
              </TabsList>
              <TabsContent value="webcam" className="space-y-3 pt-3">
                <Button className="w-full" variant="secondary" onClick={() => startWebcam()}>
                  <Camera /> Conectar câmera
                </Button>
                {devices.length > 1 && (
                  <Select
                    value={deviceId}
                    onValueChange={(v) => {
                      setDeviceId(v);
                      void startWebcam(v);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Selecionar dispositivo" />
                    </SelectTrigger>
                    <SelectContent>
                      {devices.map((d, i) => (
                        <SelectItem key={d.deviceId} value={d.deviceId}>
                          {d.label || `Câmera ${i + 1}`}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </TabsContent>
              <TabsContent value="stream" className="space-y-3 pt-3">
                <Input
                  placeholder="https://.../stream.m3u8"
                  value={streamUrl}
                  onChange={(e) => setStreamUrl(e.target.value)}
                />
                <Button className="w-full" variant="secondary" onClick={() => void startStream()}>
                  <Link2 /> Conectar stream
                </Button>
                <p className="text-xs leading-relaxed text-muted-foreground">
                  Suporta HLS (.m3u8) e MJPEG/MP4 acessíveis pelo navegador. Câmeras RTSP precisam
                  de um conversor para HLS.
                </p>
              </TabsContent>
            </Tabs>
          </div>

          <div className="rounded-2xl border border-border bg-surface p-4 shadow-panel">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
                Fotos capturadas
              </h2>
              {events.length > 0 && (
                <Button variant="ghost" size="icon" onClick={() => setEvents([])}>
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </div>
            {events.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                Cada pessoa que sai da área de entrada e chega na de saída gera uma foto aqui.
              </p>
            ) : (
              <ul className="max-h-[420px] space-y-3 overflow-y-auto pr-1">
                {events.map((ev) => (
                  <li
                    key={ev.id}
                    className={`overflow-hidden rounded-lg border ${
                      ev.status === "ignored"
                        ? "border-amber-500/40 bg-amber-500/5"
                        : "border-border bg-surface"
                    }`}
                  >
                    {ev.image ? (
                      <img
                        src={ev.image}
                        alt={`Pessoa ${ev.trackId} na passagem`}
                        className={`aspect-video w-full object-cover ${
                          ev.status === "ignored" ? "brightness-95" : ""
                        }`}
                      />
                    ) : (
                      <div className="flex aspect-video items-center justify-center text-xs text-muted-foreground">
                        sem foto disponível
                      </div>
                    )}
                    <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-xs">
                      <span className="font-mono font-semibold">#{ev.trackId}</span>
                      <span className="text-muted-foreground">
                        {new Date(ev.at).toLocaleTimeString("pt-BR")}
                      </span>
                      <span
                        className={
                          ev.status === "ignored"
                            ? "font-semibold text-amber-400"
                            : ev.status === "checking"
                              ? "text-foreground"
                              : "font-semibold text-accent"
                        }
                      >
                        {ev.status === "checking"
                          ? "analisando…"
                          : ev.status === "ignored"
                            ? "ignorada (com crachá)"
                            : "contada"}
                      </span>
                      {ev.image && (
                        <a
                          href={ev.image}
                          download={
                            ev.fileName ||
                            `${ev.status === "ignored" ? "ignorado" : "passagem"}-${ev.trackId}-${ev.at}.jpg`
                          }
                          className="text-accent hover:opacity-80"
                          title="Baixar foto para conferência"
                          aria-label="Baixar foto"
                        >
                          <Download className="h-3.5 w-3.5" />
                        </a>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
