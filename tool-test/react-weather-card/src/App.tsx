import { useCallback, useEffect, useRef, useState } from 'react'
import { CityInfo, WeatherData, describeWeather, fetchWeather, searchCity } from './api'
import './App.css'

const STORAGE_KEY = 'react-weather-card-v1'

interface Persisted {
  cities: CityInfo[]
  currentId: number | null
  unit: 'C' | 'F'
}

function loadPersisted(): Persisted {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const p = JSON.parse(raw) as Persisted
      if (Array.isArray(p.cities)) return p
    }
  } catch {
    /* 数据损坏时忽略，使用默认值 */
  }
  return { cities: [], currentId: null, unit: 'C' }
}

function cToF(c: number): number {
  return c * 9 / 5 + 32
}

export default function App() {
  const initial = useRef(loadPersisted()).current
  const [cities, setCities] = useState<CityInfo[]>(initial.cities)
  const [currentId, setCurrentId] = useState<number | null>(initial.currentId)
  const [unit, setUnit] = useState<'C' | 'F'>(initial.unit)
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [weather, setWeather] = useState<WeatherData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [leavingId, setLeavingId] = useState<number | null>(null)
  const [pulse, setPulse] = useState(0)

  const currentCity = cities.find((c) => c.id === currentId) ?? null

  // 持久化
  useEffect(() => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ cities, currentId, unit } satisfies Persisted)
    )
  }, [cities, currentId, unit])

  const loadWeather = useCallback(async (city: CityInfo) => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchWeather(city.latitude, city.longitude)
      setWeather(data)
      setPulse((p) => p + 1)
    } catch (e) {
      setWeather(null)
      setError(e instanceof Error ? e.message : '未知错误')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (currentCity) loadWeather(currentCity)
  }, [currentCity, loadWeather])

  const handleSearch = async () => {
    const name = query.trim()
    if (!name) return
    setSearching(true)
    setError(null)
    try {
      const results = await searchCity(name)
      if (results.length === 0) {
        setError(`未找到城市「${name}」，请检查名称后重试`)
        return
      }
      const first = results[0]
      setCities((prev) =>
        prev.some((c) => c.id === first.id) ? prev : [...prev, first]
      )
      setCurrentId(first.id)
      setQuery('')
    } catch (e) {
      setError(e instanceof Error ? e.message : '搜索失败')
    } finally {
      setSearching(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSearch()
  }

  const removeCity = (id: number) => {
    setLeavingId(id)
    setTimeout(() => {
      setCities((prev) => {
        const next = prev.filter((c) => c.id !== id)
        if (id === currentId) {
          setCurrentId(next.length > 0 ? next[0].id : null)
          setWeather(null)
        }
        return next
      })
      setLeavingId(null)
    }, 250)
  }

  const fmtTemp = (celsius: number): string => {
    const v = unit === 'C' ? celsius : cToF(celsius)
    return `${v.toFixed(1)}°${unit}`
  }

  const desc = weather ? describeWeather(weather.current.weather_code) : null

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌤️ 天气卡片</h1>
        <div className="unit-toggle" role="group" aria-label="温度单位">
          <button
            className={unit === 'C' ? 'active' : ''}
            onClick={() => setUnit('C')}
          >
            °C
          </button>
          <button
            className={unit === 'F' ? 'active' : ''}
            onClick={() => setUnit('F')}
          >
            °F
          </button>
        </div>
      </header>

      <div className="search-bar">
        <input
          type="text"
          placeholder="搜索城市，如：北京、上海、Tokyo..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button onClick={handleSearch} disabled={searching}>
          {searching ? '搜索中…' : '搜索'}
        </button>
      </div>

      {error && (
        <div className="error-box">
          <span>⚠️ {error}</span>
          {currentCity && (
            <button onClick={() => loadWeather(currentCity)}>重试</button>
          )}
        </div>
      )}

      <div className="layout">
        <aside className="city-list">
          <h2>城市列表</h2>
          {cities.length === 0 && <p className="empty">还没有保存的城市</p>}
          <ul>
            {cities.map((c) => (
              <li
                key={c.id}
                className={`${c.id === currentId ? 'active' : ''} ${leavingId === c.id ? 'leaving' : ''}`}
                onClick={() => setCurrentId(c.id)}
              >
                <span>{c.name}</span>
                <button
                  className="remove"
                  aria-label={`删除 ${c.name}`}
                  onClick={(e) => {
                    e.stopPropagation()
                    removeCity(c.id)
                  }}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <main className="weather-card">
          {loading && <div className="spinner" aria-label="加载中" />}
          {!loading && !currentCity && (
            <p className="empty">搜索并选择一个城市查看天气</p>
          )}
          {!loading && currentCity && weather && desc && (
            <>
              <div className="card-head">
                <h2>{currentCity.name}</h2>
                <span className="big-icon">{desc.icon}</span>
              </div>
              <div className="temp" key={pulse}>
                {fmtTemp(weather.current.temperature_2m)}
              </div>
              <div className="condition">{desc.text}</div>
              <div className="details">
                <div className="detail">
                  <span className="label">体感温度</span>
                  <span className="value">{fmtTemp(weather.current.apparent_temperature)}</span>
                </div>
                <div className="detail">
                  <span className="label">湿度</span>
                  <span className="value">{weather.current.relative_humidity_2m}%</span>
                </div>
                <div className="detail">
                  <span className="label">风速</span>
                  <span className="value">{weather.current.wind_speed_10m} km/h</span>
                </div>
              </div>
            </>
          )}
        </main>
      </div>

      <footer className="stats">
        <span>已保存城市：{cities.length}</span>
        <span>当前城市：{currentCity ? currentCity.name : '无'}</span>
        <span>
          数据更新时间：
          {weather ? new Date(weather.fetchedAt).toLocaleTimeString('zh-CN') : '—'}
        </span>
      </footer>
    </div>
  )
}
