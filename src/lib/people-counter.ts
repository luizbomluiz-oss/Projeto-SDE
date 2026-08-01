export type Box = { x: number; y: number; w: number; h: number; score: number };

export type Zone = { x: number; y: number; w: number; h: number };

export type ZoneKey = "entry" | "exit";

export type Track = {
  id: number;
  cx: number;
  cy: number;
  vx: number;
  vy: number;
  box: Box;
  missing: number;
  visitedEntry: boolean;
  visitedEntrada: boolean;
  visitedExitFirst: boolean;
  canCount: boolean;
  yInicial: number;
  counted: boolean;
  invalidDirection: boolean;
  wasInPolygon: boolean;
};

export type Point = { x: number; y: number };

export type PassEvent = {
  id: string;
  trackId: number;
  at: number;
  /** preview object URL (revoked/cleared once written to disk) */
  image: string;
  status: "ok" | "checking" | "ignored";
  reason?: string;
  savedTo?: string;
  fileName?: string;
};

export function pointInZone(z: Zone, p: Point): boolean {
  return p.x >= z.x && p.x <= z.x + z.w && p.y >= z.y && p.y <= z.y + z.h;
}

export function boxOverlapZone(b: Box, z: Zone): boolean {
  return !(b.x + b.w < z.x || b.x > z.x + z.w || b.y + b.h < z.y || b.y > z.y + z.h);
}

export function pointInPolygon(polygon: Point[], p: Point): boolean {
  if (!polygon || polygon.length < 3) return false;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersect =
      yi > p.y !== yj > p.y && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi || 0.000001) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

export function boxOverlapPolygon(b: Box, polygon: Point[]): boolean {
  if (!polygon || polygon.length === 0) return false;
  const polyMinX = Math.min(...polygon.map((p) => p.x));
  const polyMaxX = Math.max(...polygon.map((p) => p.x));
  const polyMinY = Math.min(...polygon.map((p) => p.y));
  const polyMaxY = Math.max(...polygon.map((p) => p.y));

  return !(b.x + b.w < polyMinX || b.x > polyMaxX || b.y + b.h < polyMinY || b.y > polyMaxY);
}

function ccw(a: Point, b: Point, c: Point): boolean {
  return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x);
}

function lineSegmentIntersectsSegment(p1: Point, p2: Point, p3: Point, p4: Point): boolean {
  return ccw(p1, p3, p4) !== ccw(p2, p3, p4) && ccw(p1, p2, p3) !== ccw(p1, p2, p4);
}

export function segmentIntersectsZone(p1: Point, p2: Point, z: Zone): boolean {
  if (pointInZone(z, p1) || pointInZone(z, p2)) return true;

  const tl = { x: z.x, y: z.y };
  const tr = { x: z.x + z.w, y: z.y };
  const bl = { x: z.x, y: z.y + z.h };
  const br = { x: z.x + z.w, y: z.y + z.h };

  return (
    lineSegmentIntersectsSegment(p1, p2, tl, tr) ||
    lineSegmentIntersectsSegment(p1, p2, tr, br) ||
    lineSegmentIntersectsSegment(p1, p2, br, bl) ||
    lineSegmentIntersectsSegment(p1, p2, bl, tl)
  );
}

function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

export function calculateCountWeight(box: Box): number {
  if (!box || !box.h || box.h <= 0) return 1;
  const aspectRatio = box.w / box.h;
  if (aspectRatio < 0.8) {
    return 1;
  }
  if (aspectRatio < 1.4) {
    return 2;
  }
  return 3;
}

export class PeopleTracker {
  private tracks: Track[] = [];
  private nextId = 1;
  private maxDistance: number;
  private maxMissing: number;

  constructor(opts: { maxDistance?: number; maxMissing?: number } = {}) {
    this.maxDistance = opts.maxDistance ?? 0.5;
    this.maxMissing = opts.maxMissing ?? 30;
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
  }

  getTracks(): Track[] {
    return this.tracks;
  }

  /**
   * Update tracks with normalized (0..1) detections.
   * Uses Head & Shoulders Barycenter (x + w/2, y + h*0.20), Adaptive Grace Period, Strict 80px Centroid Pairing, and Polygon ROI (or fallback rectangle).
   */
  update(
    detections: Box[],
    zones: { entry: Zone; exit: Zone },
    exitPolygon?: Point[],
  ): { passes: { trackId: number; countWeight: number }[]; tracks: Track[] } {
    const fallbackPoly: Point[] = [
      { x: zones.exit.x, y: zones.exit.y },
      { x: zones.exit.x + zones.exit.w, y: zones.exit.y },
      { x: zones.exit.x + zones.exit.w, y: zones.exit.y + zones.exit.h },
      { x: zones.exit.x, y: zones.exit.y + zones.exit.h },
    ];
    const activePolygon = exitPolygon && exitPolygon.length >= 3 ? exitPolygon : fallbackPoly;

    const unmatched = new Set(detections.map((_, i) => i));
    const pairs: { t: number; d: number; dist: number }[] = [];
    const exitTop = Math.min(...activePolygon.map((p) => p.y));
    const exitBottom = Math.max(...activePolygon.map((p) => p.y));

    // Head & Shoulders Barycenter Euclidean distance (in px using 640x480 ref scale)
    this.tracks.forEach((track, ti) => {
      detections.forEach((det, di) => {
        const headX = det.x + det.w / 2;
        const headY = det.y + det.h * 0.2;
        const distPx = Math.hypot((headX - track.cx) * 640, (headY - track.cy) * 480);

        // Strict centroid pairing threshold: <= 80px
        if (distPx <= 80) {
          pairs.push({ t: ti, d: di, dist: distPx });
        }
      });
    });

    pairs.sort((p, q) => p.dist - q.dist);

    const usedTracks = new Set<number>();
    const passes: { trackId: number; countWeight: number }[] = [];

    for (const pair of pairs) {
      if (usedTracks.has(pair.t) || !unmatched.has(pair.d)) continue;
      usedTracks.add(pair.t);
      unmatched.delete(pair.d);

      const det = detections[pair.d];
      const track = this.tracks[pair.t];

      // 1. Head & Shoulders Barycenter: X = x + (w/2), Y = y + (h * 0.20)
      const headX = det.x + det.w / 2;
      const headY = det.y + det.h * 0.2;
      const headPoint: Point = { x: headX, y: headY };

      track.cx = headX;
      track.cy = headY;
      track.box = det;
      track.missing = 0;
      track.vx = 0;
      track.vy = 0;

      // 2. Vector Direction Validation (Delta Y in px):
      const deltaYPx = (headY - track.yInicial) * 480;

      // Cancel permission if initial position was strictly below exit region and movement is upward (< -15px)
      if (track.yInicial > exitBottom && deltaYPx < -15) {
        track.canCount = false;
      } else if (deltaYPx >= 0 || track.yInicial <= exitBottom) {
        track.canCount = true;
      }

      // 3. Point collision with Exit Zone (Polygon ROI)
      const touchesExit = pointInPolygon(activePolygon, headPoint);

      // 1. Record polygon entry moment & 3. Reset state outside polygon
      if (touchesExit) {
        if (!track.wasInPolygon) {
          track.wasInPolygon = true;
          // First time entering polygon: check vertical displacement
          if (deltaYPx < 15) {
            track.invalidDirection = true;
          }
        }
      } else {
        track.wasInPolygon = false;
        // Reset state if person completely exits polygon and moves to upper part of screen (headY < exitTop)
        if (headY < exitTop) {
          track.invalidDirection = false;
        }
      }

      // 2. Strict Trigger Validation: touchesExit && canCount && !counted && !invalidDirection && deltaYPx >= 15
      if (
        touchesExit &&
        track.canCount &&
        !track.counted &&
        !track.invalidDirection &&
        deltaYPx >= 15
      ) {
        track.counted = true;
        const countWeight = calculateCountWeight(det);
        passes.push({ trackId: track.id, countWeight });
      }
    }

    // Process new detections (> 80px from any existing track)
    for (const di of unmatched) {
      const det = detections[di];
      const headX = det.x + det.w / 2;
      const headY = det.y + det.h * 0.2;
      const headPoint: Point = { x: headX, y: headY };

      // Origin Trajectory Registration: Record initial barycenter Y position
      const yInicial = headY;
      const deltaYPx = (headY - yInicial) * 480; // 0 for initial frame

      // Margem de Tolerância de Origem: canCount = true se yInicial <= limite INFERIOR do polígono de saída
      const canCount = yInicial <= exitBottom;

      const id = this.nextId++;
      const inExit = pointInPolygon(activePolygon, headPoint);

      let invalidDirection = false;
      if (inExit) {
        // First detected inside polygon: deltaYPx is 0 (< 15)
        invalidDirection = true;
      }

      let counted = false;

      // Strict Trigger Validation for new track
      if (inExit && canCount && !counted && !invalidDirection && deltaYPx >= 15) {
        counted = true;
        const countWeight = calculateCountWeight(det);
        passes.push({ trackId: id, countWeight });
      }

      const track: Track = {
        id,
        cx: headX,
        cy: headY,
        vx: 0,
        vy: 0,
        box: det,
        missing: 0,
        visitedEntry: canCount,
        visitedEntrada: canCount,
        visitedExitFirst: !canCount,
        canCount,
        yInicial,
        counted,
        invalidDirection,
        wasInPolygon: inExit,
      };

      this.tracks.push(track);
    }

    // Adaptive Grace Period / Border Tracker Retention
    // Keep tracks alive during missing frames: 300 frames (~10s) inside/on border of exit polygon, 30 frames (~1s) outside
    this.tracks = this.tracks.filter((t, ti) => {
      if (usedTracks.has(ti)) return true;
      t.missing += 1;

      const headPoint: Point = { x: t.cx, y: t.cy };
      const inExitOrBorder =
        pointInPolygon(activePolygon, headPoint) || boxOverlapPolygon(t.box, activePolygon);
      const maxAllowedMissing = inExitOrBorder ? 300 : 30;

      return t.missing <= maxAllowedMissing;
    });

    return { passes, tracks: this.tracks };
  }
}
