export interface CityInfo {
  id: number
  name: string
  latitude: number
  longitude: number
}

export interface CurrentWeather {
  temperature_2m: number
  apparent_temperature: number
  relative_humidity_2m: number
  wind_speed_10m: number
  weather_code: number
}

export interface WeatherData {
  current: CurrentWeather
  fetchedAt: number
}

const GEO_URL = 'https://geocoding-api.open-meteo.com/v1/search'
const WEATHER_URL = 'https://api.open-meteo.com/v1/forecast'

// weather_code -> 中文描述 + 图标
export const WEATHER_CODE_MAP: Record<number, { text: string; icon: string }> = {
  0: { text: '晴', icon: '☀️' },
  1: { text: '大致晴朗', icon: '🌤️' },
  2: { text: '局部多云', icon: '⛅' },
  3: { text: '阴', icon: '☁️' },
  45: { text: '雾', icon: '🌫️' },
  48: { text: '雾凇', icon: '🌫️' },
  51: { text: '小毛毛雨', icon: '🌦️' },
  53: { text: '毛毛雨', icon: '🌦️' },
  55: { text: '大毛毛雨', icon: '🌧️' },
  61: { text: '小雨', icon: '🌦️' },
  63: { text: '中雨', icon: '🌧️' },
  65: { text: '大雨', icon: '🌧️' },
  71: { text: '小雪', icon: '🌨️' },
  73: { text: '中雪', icon: '❄️' },
  75: { text: '大雪', icon: '❄️' },
  77: { text: '雪粒', icon: '🌨️' },
  80: { text: '小阵雨', icon: '🌦️' },
  81: { text: '阵雨', icon: '🌧️' },
  82: { text: '强阵雨', icon: '⛈️' },
  85: { text: '小阵雪', icon: '🌨️' },
  86: { text: '大阵雪', icon: '❄️' },
  95: { text: '雷暴', icon: '⛈️' },
  96: { text: '雷暴伴冰雹', icon: '⛈️' },
  99: { text: '强雷暴伴冰雹', icon: '⛈️' },
}

export function describeWeather(code: number): { text: string; icon: string } {
  return WEATHER_CODE_MAP[code] ?? { text: '未知天气', icon: '🌡️' }
}

export async function searchCity(name: string): Promise<CityInfo[]> {
  const resp = await fetch(
    `${GEO_URL}?name=${encodeURIComponent(name)}&count=5&language=zh`
  )
  if (!resp.ok) throw new Error(`地理编码请求失败（HTTP ${resp.status}）`)
  const data = await resp.json()
  if (!Array.isArray(data.results)) return []
  return data.results.map((r: any) => ({
    id: r.id,
    name: r.name,
    latitude: r.latitude,
    longitude: r.longitude,
  }))
}

export async function fetchWeather(lat: number, lon: number): Promise<WeatherData> {
  const resp = await fetch(
    `${WEATHER_URL}?latitude=${lat}&longitude=${lon}` +
      `&current=temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code`
  )
  if (!resp.ok) throw new Error(`天气请求失败（HTTP ${resp.status}）`)
  const data = await resp.json()
  if (!data.current) throw new Error('天气接口返回数据格式异常：缺少 current 字段')
  return { current: data.current as CurrentWeather, fetchedAt: Date.now() }
}
