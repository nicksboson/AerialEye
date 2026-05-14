import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MapContainer, TileLayer, CircleMarker, Popup, useMap, ZoomControl, Polygon } from 'react-leaflet'
import { useNavigate } from 'react-router-dom'
import L from 'leaflet'
import {
  Layers,
  Maximize2,
  Minimize2,
  Search,
  SatelliteDish,
  Map as MapIcon,
  Download,
  X,
  Check,
  MapPin,
  AlertTriangle,
} from 'lucide-react'
import { getLatestAnalysis } from '../lib/analysis-storage'
import type { AnalysisApiSuccess, LeafletMarker, LeafletPolygon } from '../types/analysis'
import { exportGeoJSON } from '../utils/api'
import 'leaflet/dist/leaflet.css'

delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

const BASE_LAYERS = {
  satellite: {
    label: 'Satellite',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '(c) Esri, DigitalGlobe',
  },
  street: {
    label: 'Street',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '(c) OpenStreetMap contributors',
  },
} as const

type BaseKey = keyof typeof BASE_LAYERS

function MapResizer({ trigger }: { trigger: unknown }) {
  const map = useMap()
  useEffect(() => {
    const timer = setTimeout(() => map.invalidateSize(), 300)
    return () => clearTimeout(timer)
  }, [trigger, map])
  return null
}

function countByLayer(polygons: LeafletPolygon[]): Record<string, number> {
  return polygons.reduce<Record<string, number>>((accumulator, polygon) => {
    accumulator[polygon.layer_id] = (accumulator[polygon.layer_id] ?? 0) + 1
    return accumulator
  }, {})
}

export default function GIS() {
  const navigate = useNavigate()
  const [activeBase, setActiveBase] = useState<BaseKey>('satellite')
  const [analysisResult] = useState<AnalysisApiSuccess | null>(() => getLatestAnalysis())
  const [enabledLayers, setEnabledLayers] = useState<string[]>([])
  const [layerMenuOpen, setLayerMenuOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [exported, setExported] = useState(false)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const layerMenuRef = useRef<HTMLDivElement>(null)

  const mapCenter = analysisResult?.transformed.gis_mapping.map_center ?? { lat: 28.6139, lng: 77.209 }
  const polygons = analysisResult?.transformed.gis_mapping.leaflet_polygons ?? []
  const markers = analysisResult?.transformed.gis_mapping.leaflet_markers ?? []
  const layerDefinitions = analysisResult?.transformed.chart_data.category_comparison_data ?? []

  useEffect(() => {
    const allLayerIds = layerDefinitions.map((item) => item.layer_id).filter(Boolean)
    setEnabledLayers(allLayerIds)
  }, [layerDefinitions])

  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (layerMenuRef.current && !layerMenuRef.current.contains(event.target as Node)) {
        setLayerMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const toggleFullscreen = useCallback(async () => {
    if (!document.fullscreenElement) {
      await wrapperRef.current?.requestFullscreen()
      return
    }
    await document.exitFullscreen()
  }, [])

  useEffect(() => {
    const handler = () => setIsFullscreen(Boolean(document.fullscreenElement))
    document.addEventListener('fullscreenchange', handler)
    return () => document.removeEventListener('fullscreenchange', handler)
  }, [])

  const filteredLayerDefinitions = useMemo(
    () =>
      layerDefinitions.filter((item) =>
        item.category.toLowerCase().includes(searchQuery.trim().toLowerCase()),
      ),
    [layerDefinitions, searchQuery],
  )

  const visiblePolygons = useMemo(
    () => polygons.filter((polygon) => enabledLayers.includes(polygon.layer_id)),
    [polygons, enabledLayers],
  )

  const visibleMarkers = useMemo(
    () => markers.filter((marker) => enabledLayers.includes(marker.layer_id)),
    [markers, enabledLayers],
  )

  const layerCountMap = useMemo(() => countByLayer(visiblePolygons), [visiblePolygons])

  const toggleLayer = (id: string) => {
    setEnabledLayers((previous) =>
      previous.includes(id) ? previous.filter((value) => value !== id) : [...previous, id],
    )
  }

  const handleExport = async () => {
    if (!analysisResult) return
    const blob = await exportGeoJSON(analysisResult)
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'spatial_assets.geojson'
    anchor.click()
    URL.revokeObjectURL(url)
    setExported(true)
    setTimeout(() => setExported(false), 1200)
  }

  if (!analysisResult) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="min-h-screen bg-[#fafaf9] pt-20 px-4"
      >
        <div className="max-w-3xl mx-auto bg-white rounded-2xl border border-stone-200 shadow-soft p-8">
          <div className="flex items-start gap-3 text-amber-800 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold text-sm">No live analysis found</p>
              <p className="text-sm text-amber-700 mt-1">
                Run an image detection first so GIS overlays can be generated from AI output.
              </p>
            </div>
          </div>
          <div className="mt-6">
            <button
              onClick={() => navigate('/detect')}
              className="px-4 py-2.5 rounded-xl bg-teal-600 text-white text-sm font-semibold hover:bg-teal-700 transition-colors"
            >
              Go To Detection
            </button>
          </div>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="bg-[#fafaf9] pt-16 h-screen flex flex-col overflow-hidden"
    >
      <div className="bg-white border-b border-stone-200 px-4 sm:px-6 py-2.5 flex items-center justify-between gap-3 shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-teal-500" />
            <span className="text-sm font-semibold text-stone-800">GIS Asset Map</span>
          </div>
          <span className="hidden sm:block w-px h-4 bg-stone-200" />
          <span className="hidden sm:block text-xs text-stone-400 font-mono">
            {mapCenter.lat.toFixed(4)} deg N | {mapCenter.lng.toFixed(4)} deg E
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-stone-500">
          <MapPin className="w-3 h-3 text-teal-500" />
          <span className="font-medium">{visiblePolygons.length} assets visible</span>
        </div>
      </div>

      <div ref={wrapperRef} className="flex-1 relative overflow-hidden">
        <div className="absolute top-3 left-3 right-3 z-[500] flex items-center justify-between gap-2 pointer-events-none">
          <div className="flex items-center gap-2 pointer-events-auto">
            <div className="relative" ref={layerMenuRef}>
              <button
                onClick={() => setLayerMenuOpen((value) => !value)}
                className={`h-9 px-3.5 rounded-xl inline-flex items-center gap-2 text-sm font-semibold shadow-medium border transition-all duration-150 ${
                  layerMenuOpen
                    ? 'bg-teal-600 text-white border-teal-600 shadow-teal-200'
                    : 'bg-white/96 backdrop-blur-md text-stone-700 border-stone-200/80 hover:border-stone-300'
                }`}
              >
                <Layers className="w-4 h-4" />
                <span className="hidden sm:inline">Layers</span>
                {!layerMenuOpen && (
                  <span className="w-5 h-5 rounded-full bg-teal-100 text-teal-700 text-[10px] font-bold inline-flex items-center justify-center">
                    {enabledLayers.length}
                  </span>
                )}
              </button>

              <AnimatePresence>
                {layerMenuOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -6, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.97 }}
                    transition={{ duration: 0.15, ease: 'easeOut' }}
                    className="absolute left-0 top-11 w-64 bg-white rounded-2xl shadow-large border border-stone-200/80 overflow-hidden z-30"
                  >
                    <div className="px-4 pt-3.5 pb-2.5 border-b border-stone-100 flex items-center justify-between">
                      <span className="text-xs font-semibold text-stone-500 uppercase tracking-wider flex items-center gap-1.5">
                        <Layers className="w-3.5 h-3.5" />
                        Asset Layers
                      </span>
                      <button
                        onClick={() => setLayerMenuOpen(false)}
                        className="w-5 h-5 flex items-center justify-center text-stone-400 hover:text-stone-700 transition-colors"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <div className="px-3 py-2.5 border-b border-stone-100">
                      <div className="relative">
                        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-stone-400 pointer-events-none" />
                        <input
                          type="text"
                          placeholder="Search layers..."
                          value={searchQuery}
                          onChange={(event) => setSearchQuery(event.target.value)}
                          className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-stone-50 border border-stone-200 text-xs text-stone-800 placeholder-stone-400 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-100 transition-all"
                        />
                      </div>
                    </div>

                    <div className="px-2 py-2 flex flex-col gap-0.5 max-h-72 overflow-y-auto">
                      {filteredLayerDefinitions.map((layer) => {
                        const active = enabledLayers.includes(layer.layer_id)
                        const count = layerCountMap[layer.layer_id] ?? 0
                        return (
                          <button
                            key={layer.layer_id}
                            onClick={() => toggleLayer(layer.layer_id)}
                            className={`w-full flex items-center justify-between px-2.5 py-2 rounded-xl text-left transition-all duration-150 ${
                              active ? 'bg-stone-50 hover:bg-stone-100' : 'opacity-45 hover:opacity-65 hover:bg-stone-50'
                            }`}
                          >
                            <div className="flex items-center gap-2.5 min-w-0">
                              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: layer.color }} />
                              <span className="text-xs font-semibold text-stone-800 truncate">{layer.category}</span>
                            </div>
                            <div className="flex items-center gap-1.5 shrink-0">
                              {count > 0 && <span className="text-[10px] text-stone-400 font-mono tabular-nums">{count}</span>}
                              <div className={`relative w-8 h-4 rounded-full p-0.5 transition-colors ${active ? 'bg-teal-500' : 'bg-stone-200'}`}>
                                <motion.div
                                  className="w-3 h-3 rounded-full bg-white shadow-sm"
                                  animate={{ x: active ? 16 : 0 }}
                                  transition={{ type: 'spring', stiffness: 600, damping: 35 }}
                                />
                              </div>
                            </div>
                          </button>
                        )
                      })}
                    </div>

                    <div className="px-3 pb-3 pt-2 border-t border-stone-100 flex flex-col gap-2">
                      <div className="flex gap-2">
                        <button
                          onClick={() => setEnabledLayers(layerDefinitions.map((item) => item.layer_id))}
                          className="flex-1 py-1.5 rounded-lg text-xs font-semibold text-teal-700 bg-teal-50 hover:bg-teal-100 border border-teal-200 transition-colors"
                        >
                          All on
                        </button>
                        <button
                          onClick={() => setEnabledLayers([])}
                          className="flex-1 py-1.5 rounded-lg text-xs font-semibold text-stone-600 bg-stone-50 hover:bg-stone-100 border border-stone-200 transition-colors"
                        >
                          All off
                        </button>
                      </div>
                      <button
                        onClick={() => void handleExport()}
                        className={`w-full py-2 rounded-xl text-xs font-semibold inline-flex items-center justify-center gap-1.5 transition-all duration-200 ${
                          exported ? 'bg-teal-50 text-teal-700 border border-teal-200' : 'bg-teal-600 text-white hover:bg-teal-700 active:scale-[0.98]'
                        }`}
                      >
                        {exported ? <Check className="w-3.5 h-3.5" /> : <Download className="w-3.5 h-3.5" />}
                        {exported ? 'Exported!' : 'Export GeoJSON'}
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="flex items-center bg-white/96 backdrop-blur-md rounded-xl border border-stone-200/80 shadow-medium overflow-hidden h-9">
              {(Object.entries(BASE_LAYERS) as [BaseKey, (typeof BASE_LAYERS)[BaseKey]][]).map(([key, val]) => (
                <button
                  key={key}
                  onClick={() => setActiveBase(key)}
                  className={`h-full px-3.5 inline-flex items-center gap-1.5 text-xs font-semibold transition-colors ${
                    activeBase === key ? 'bg-teal-600 text-white' : 'text-stone-600 hover:bg-stone-50 hover:text-stone-900'
                  }`}
                >
                  {key === 'satellite' ? <SatelliteDish className="w-3.5 h-3.5" /> : <MapIcon className="w-3.5 h-3.5" />}
                  <span className="hidden sm:inline">{val.label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="pointer-events-auto">
            <button
              onClick={() => void toggleFullscreen()}
              aria-label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
              className="w-9 h-9 bg-white/96 backdrop-blur-md rounded-xl flex items-center justify-center text-stone-600 hover:text-teal-700 shadow-medium border border-stone-200/80 transition-colors"
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          </div>
        </div>

        <MapContainer
          center={[mapCenter.lat, mapCenter.lng]}
          zoom={15}
          className="w-full h-full"
          zoomControl={false}
          attributionControl={true}
        >
          <MapResizer trigger={isFullscreen} />
          <ZoomControl position="bottomright" />

          <TileLayer key={activeBase} url={BASE_LAYERS[activeBase].url} attribution={BASE_LAYERS[activeBase].attribution} maxZoom={20} />

          {visiblePolygons.map((polygon) => (
            <Polygon
              key={polygon.id}
              positions={polygon.coordinates}
              pathOptions={{
                color: polygon.color,
                weight: 2,
                fillColor: polygon.color,
                fillOpacity: 0.28,
              }}
            >
              <Popup offset={[0, -4]} closeButton={false}>
                <div className="min-w-[180px]">
                  <div className="flex items-start gap-2 mb-2.5">
                    <span className="w-3 h-3 rounded-full shrink-0 mt-0.5 ring-2 ring-white shadow-sm" style={{ backgroundColor: polygon.color }} />
                    <div>
                      <p className="font-semibold text-stone-900 text-sm leading-snug">{polygon.category}</p>
                      <p className="text-[11px] text-stone-400 mt-0.5">{polygon.subcategory}</p>
                    </div>
                  </div>
                  <div className="flex flex-col gap-0.5 text-[11px] text-stone-500">
                    <p>
                      <span className="text-stone-400">Confidence </span>
                      {polygon.confidence_percent.toFixed(1)}%
                    </p>
                    <p>
                      <span className="text-stone-400">Priority </span>
                      {polygon.maintenance_priority}
                    </p>
                    <p>
                      <span className="text-stone-400">Condition </span>
                      {polygon.condition_status}
                    </p>
                  </div>
                </div>
              </Popup>
            </Polygon>
          ))}

          {visibleMarkers.map((marker: LeafletMarker) => (
            <CircleMarker
              key={marker.id}
              center={[marker.lat, marker.lng]}
              radius={5}
              pathOptions={{
                color: 'white',
                weight: 1.2,
                fillColor: marker.color,
                fillOpacity: 0.92,
              }}
            >
              <Popup offset={[0, -4]} closeButton={false}>
                <div className="min-w-[170px]">
                  <p className="font-semibold text-stone-900 text-sm">{marker.category}</p>
                  <p className="text-[11px] text-stone-500 mt-1">{marker.label}</p>
                  <p className="text-[11px] text-stone-500 mt-1">Confidence: {marker.confidence_percent.toFixed(1)}%</p>
                  <p className="text-[11px] text-stone-500">Area: {marker.estimated_area_sq_m.toFixed(1)} sq.m</p>
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>

        {enabledLayers.length > 0 && (
          <div className="absolute bottom-10 left-3 z-[400] bg-white/95 backdrop-blur-md rounded-xl shadow-medium border border-stone-200/80 overflow-hidden max-w-[220px]">
            <div className="px-3 py-2 flex flex-col gap-1.5">
              {layerDefinitions
                .filter((item) => enabledLayers.includes(item.layer_id))
                .map((item) => (
                  <div key={item.layer_id} className="flex items-center gap-2 text-[11px]">
                    <span className="w-2.5 h-2.5 rounded-full shrink-0 ring-1 ring-white" style={{ backgroundColor: item.color }} />
                    <span className="text-stone-700 font-medium truncate leading-none">{item.category}</span>
                    <span className="ml-auto pl-1 text-stone-400 tabular-nums font-mono text-[10px]">
                      {layerCountMap[item.layer_id] ?? 0}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  )
}
