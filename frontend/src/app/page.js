"use client"

import { useState, useRef, useEffect } from "react"

const GATEWAY_URL = "http://127.0.0.1:5000"

function Icon({ name, className = "" }) {
  return <span className={`material-symbols-rounded ${className}`}>{name}</span>
}

export default function StyleFinder() {
  const [imagePreview, setImagePreview] = useState(null)
  const [query, setQuery] = useState("")
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)
  const scrollRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages, loading])

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const base64 = reader.result.split(",")[1]
      setImagePreview({ dataUrl: reader.result, base64 })
    }
    reader.readAsDataURL(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (!file) return
    handleFileChange({ target: { files: [file] } })
  }

  const removeImage = () => {
    setImagePreview(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleSubmit = async () => {
    if (!query.trim()) {
      setError("Type what you're looking for first.")
      return
    }
    setError(null)
    setLoading(true)

    const userTurn = { role: "user", text: query, image: imagePreview?.dataUrl }
    setMessages((prev) => [...prev, userTurn])
    setImagePreview(null)
    if (fileInputRef.current) fileInputRef.current.value = ""

    try {
      const res = await fetch(`${GATEWAY_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: userTurn.image ? userTurn.image.split(",")[1] : null,
          query: query,
          session_id: sessionId,
        }),
      })
      if (!res.ok) throw new Error(`Gateway returned ${res.status}`)
      const data = await res.json()
      setSessionId(data.session_id)
      setMessages((prev) => [...prev, { role: "assistant", text: data.message, matches: data.matches }])
    } catch (err) {
      setError("Couldn't reach the style finder. Is the gateway running on port 5000?")
    } finally {
      setLoading(false)
      setQuery("")
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const autoGrow = (e) => {
    setQuery(e.target.value)
    e.target.style.height = "auto"
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px"
  }

  function formatMessage(text) {
  // split on lines first, so we can detect bullets per line
  const lines = text.split("\n")

  return lines.map((line, i) => {
    const trimmed = line.trim()
    const isBullet = trimmed.startsWith("* ") && !trimmed.startsWith("**")

    const content = isBullet ? trimmed.slice(2) : line
    const parts = renderBold(content)

    if (isBullet) {
      return (
        <div key={i} className="flex gap-2 pl-1">
          <span className="text-[#9B9488]">•</span>
          <span>{parts}</span>
        </div>
      )
    }

    return <div key={i}>{parts}</div>
  })
}

function renderBold(text) {
  const segments = text.split(/(\*\*[^*]+\*\*)/g)
  return segments.map((seg, i) => {
    if (seg.startsWith("**") && seg.endsWith("**")) {
      return <strong key={i} className="font-semibold">{seg.slice(2, -2)}</strong>
    }
    return seg
  })
}

  return (
    <main className="h-screen flex flex-col bg-[#FAFAF8]">
      {/* Top bar */}
      <header className="shrink-0 border-b border-[#ECEAE4] bg-white/80 backdrop-blur-sm px-6 py-4">
        <div className="max-w-2xl mx-auto flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-full bg-[#1F1B16] flex items-center justify-center">
            <Icon name="styler" className="text-white text-[18px]" />
          </div>
          <div>
            <h1 className="text-[15px] font-semibold text-[#1F1B16] leading-tight">Style Finder</h1>
            <p className="text-[12px] text-[#9B9488] leading-tight">Upload a piece, describe the vibe</p>
          </div>
        </div>
      </header>

      {/* Scrollable chat */}
    <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
      <div className="max-w-2xl mx-auto flex flex-col gap-5 min-h-full">
        {messages.length === 0 && (
              <div className="flex-1 flex flex-col items-center justify-center text-center gap-3">
                <div className="w-14 h-14 rounded-full bg-[#EFEDE6] flex items-center justify-center">
                  <Icon name="checkroom" className="text-[#9B9488] text-[26px]" />
                </div>
                <p className="text-[14px] text-[#9B9488] max-w-xs">
                  Attach a photo and tell me what occasion you're styling for.
                </p>
              </div>
            )}

          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[82%] rounded-[20px] px-4 py-3 text-[14px] leading-relaxed whitespace-pre-wrap ${
                  m.role === "user"
                    ? "bg-[#1F1B16] text-[#FAFAF8] rounded-br-md"
                    : "bg-white border border-[#ECEAE4] text-[#2A261F] rounded-bl-md"
                }`}
              >
                {m.image && (
                  <img src={m.image} alt="uploaded" className="rounded-[14px] mb-2 max-h-44 object-cover" />
                )}
                <div className="space-y-1">{formatMessage(m.text)}</div>


                {m.matches && m.matches.length > 0 && (
                  <div className="flex gap-2.5 mt-3 overflow-x-auto pb-1 -mx-1 px-1">
                    {m.matches.map((match, idx) => (
                      <div key={idx} className="shrink-0 w-[88px]">
                        <img
                          src={match.url}
                          alt={match.caption}
                          className="w-[88px] h-[88px] object-cover rounded-[12px] border border-[#ECEAE4]"
                        />
                        <p className="text-[10px] text-[#9B9488] mt-1 text-center">
                          {Math.round(match.similarity * 100)}% match
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-[#ECEAE4] rounded-[20px] rounded-bl-md px-4 py-3 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#C9C3B6] animate-bounce [animation-delay:-0.3s]" />
                <span className="w-1.5 h-1.5 rounded-full bg-[#C9C3B6] animate-bounce [animation-delay:-0.15s]" />
                <span className="w-1.5 h-1.5 rounded-full bg-[#C9C3B6] animate-bounce" />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Fixed input bar */}
      <div className="shrink-0 border-t border-[#ECEAE4] bg-white px-6 py-4">
        <div className="max-w-2xl mx-auto">
          {error && (
            <div className="flex items-center gap-1.5 text-[12px] text-[#9A4B3A] bg-[#F6E9E4] rounded-full px-3 py-1.5 mb-2 w-fit">
              <Icon name="error" className="text-[14px]" />
              {error}
            </div>
          )}

          {imagePreview && (
            <div className="relative inline-block mb-2">
              <img src={imagePreview.dataUrl} alt="preview" className="h-16 w-16 rounded-[12px] object-cover border border-[#ECEAE4]" />
              <button
                onClick={removeImage}
                className="absolute -top-1.5 -right-1.5 bg-[#1F1B16] text-white rounded-full w-5 h-5 flex items-center justify-center"
              >
                <Icon name="close" className="text-[12px]" />
              </button>
            </div>
          )}

          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className="flex items-end gap-2 bg-[#F2F0EA] rounded-[28px] px-2 py-2"
          >
            <button
              onClick={() => fileInputRef.current.click()}
              className="shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-[#6B6256] hover:bg-[#E6E3DA] transition"
              title="Attach image"
            >
              <Icon name="add_photo_alternate" className="text-[20px]" />
            </button>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />

            <textarea
              ref={textareaRef}
              value={query}
              onChange={autoGrow}
              onKeyDown={handleKeyDown}
              placeholder="Find me something casual for a beach day…"
              rows={1}
              className="flex-1 resize-none outline-none text-[14px] py-2 bg-transparent text-[#1F1B16] placeholder:text-[#A39C8F] max-h-[120px]"
            />

            <button
              onClick={handleSubmit}
              disabled={loading || !query.trim()}
              suppressHydrationWarning
              className="shrink-0 w-9 h-9 rounded-full bg-[#1F1B16] text-white flex items-center justify-center disabled:opacity-30 hover:opacity-90 transition"
            >
              <Icon name="arrow_upward" className="text-[18px]" />
            </button>
          </div>
        </div>
      </div>
    </main>
  )
}