import { useState, useEffect } from 'react'
import './App.css'

interface Todo {
  id: number
  text: string
  completed: boolean
  createdAt: number
}

type Filter = 'all' | 'active' | 'completed'

const STORAGE_KEY = 'react-todo-app-todos'

function App() {
  const [todos, setTodos] = useState<Todo[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })
  const [input, setInput] = useState('')
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editingText, setEditingText] = useState('')
  const [filter, setFilter] = useState<Filter>('all')

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(todos))
  }, [todos])

  const addTodo = () => {
    const text = input.trim()
    if (!text) return
    setTodos((prev) => [
      { id: Date.now(), text, completed: false, createdAt: Date.now() },
      ...prev,
    ])
    setInput('')
  }

  const deleteTodo = (id: number) => {
    setTodos((prev) => prev.filter((t) => t.id !== id))
  }

  const toggleTodo = (id: number) => {
    setTodos((prev) =>
      prev.map((t) => (t.id === id ? { ...t, completed: !t.completed } : t)),
    )
  }

  const startEdit = (todo: Todo) => {
    setEditingId(todo.id)
    setEditingText(todo.text)
  }

  const saveEdit = () => {
    const text = editingText.trim()
    if (editingId !== null && text) {
      setTodos((prev) =>
        prev.map((t) => (t.id === editingId ? { ...t, text } : t)),
      )
    }
    setEditingId(null)
  }

  const filtered = todos.filter((t) =>
    filter === 'all' ? true : filter === 'active' ? !t.completed : t.completed,
  )

  const total = todos.length
  const doneCount = todos.filter((t) => t.completed).length
  const activeCount = total - doneCount

  return (
    <div className="app">
      <div className="card">
        <h1 className="title">📝 我的待办清单</h1>

        <div className="input-row">
          <input
            className="todo-input"
            value={input}
            placeholder="输入新的待办事项..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && addTodo()}
          />
          <button className="add-btn" onClick={addTodo}>
            添加
          </button>
        </div>

        <div className="filters">
          {(
            [
              ['all', `全部 (${total})`],
              ['active', `进行中 (${activeCount})`],
              ['completed', `已完成 (${doneCount})`],
            ] as [Filter, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              className={`filter-btn ${filter === key ? 'active' : ''}`}
              onClick={() => setFilter(key)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="stats">
          共 {total} 项 · 进行中 {activeCount} 项 · 已完成 {doneCount} 项 ·
          完成率 {total ? Math.round((doneCount / total) * 100) : 0}%
        </div>

        <ul className="todo-list">
          {filtered.map((todo) => (
            <li key={todo.id} className={`todo-item ${todo.completed ? 'completed' : ''}`}>
              <input
                type="checkbox"
                className="checkbox"
                checked={todo.completed}
                onChange={() => toggleTodo(todo.id)}
              />
              {editingId === todo.id ? (
                <input
                  className="edit-input"
                  value={editingText}
                  autoFocus
                  onChange={(e) => setEditingText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') saveEdit()
                    if (e.key === 'Escape') setEditingId(null)
                  }}
                  onBlur={saveEdit}
                />
              ) : (
                <span className="text" onDoubleClick={() => startEdit(todo)}>
                  {todo.text}
                </span>
              )}
              <div className="actions">
                <button className="btn edit" onClick={() => startEdit(todo)} title="编辑">
                  ✏️
                </button>
                <button className="btn delete" onClick={() => deleteTodo(todo.id)} title="删除">
                  🗑️
                </button>
              </div>
            </li>
          ))}
          {filtered.length === 0 && <li className="empty">暂无待办事项 ✨</li>}
        </ul>
      </div>
    </div>
  )
}

export default App
