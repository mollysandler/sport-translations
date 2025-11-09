"use client"

import { useState } from "react"
import "./FeedbackPanel.css"

export default function FeedbackPanel({ sourceLanguage, targetLanguage, audioInput }) {
  const [rating, setRating] = useState(0)
  const [feedbackType, setFeedbackType] = useState("general")
  const [comments, setComments] = useState("")
  const [submitted, setSubmitted] = useState(false)

  const feedbackTypes = [
    { value: "accuracy", label: "Translation Accuracy" },
    { value: "clarity", label: "Audio Clarity" },
    { value: "timing", label: "Timing/Sync" },
    { value: "voice", label: "Voice Quality" },
    { value: "general", label: "General Feedback" },
  ]

  const handleSubmit = (e) => {
    e.preventDefault()
    console.log("[v0] Feedback submitted:", {
      rating,
      feedbackType,
      comments,
      sourceLanguage,
      targetLanguage,
      audioInput: audioInput.name,
    })
    setSubmitted(true)
    setTimeout(() => {
      setRating(0)
      setFeedbackType("general")
      setComments("")
      setSubmitted(false)
    }, 2000)
  }

  return (
    <div className="feedback-card">
      <h3>Send Feedback</h3>

      {submitted ? (
        <div className="feedback-success">
          <span className="success-icon">✓</span>
          <p>Thank you for your feedback!</p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="feedback-form">
          <div className="form-group">
            <label>Rate Your Experience:</label>
            <div className="star-rating">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  type="button"
                  className={`star ${rating >= star ? "filled" : ""}`}
                  onClick={() => setRating(star)}
                  title={star + " stars"}
                >
                  ★
                </button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="feedback-type">Feedback Type:</label>
            <select
              id="feedback-type"
              value={feedbackType}
              onChange={(e) => setFeedbackType(e.target.value)}
              className="feedback-select"
            >
              {feedbackTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="comments">Comments (optional):</label>
            <textarea
              id="comments"
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              placeholder="Tell us what you think..."
              className="feedback-textarea"
              rows="4"
            />
          </div>

          <button type="submit" className="submit-feedback-button">
            Submit Feedback
          </button>
        </form>
      )}
    </div>
  )
}
