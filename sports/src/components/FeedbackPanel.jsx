"use client"

import { useState } from "react"
import "./FeedbackPanel.css"

const BASE_URL =
  "https://mollysandler--sports-translation-api-translatorservice-f-6a7378.modal.run";

export default function FeedbackPanel({ sourceLanguage, targetLanguage, audioInput, showToast }) {
  const [rating, setRating] = useState(0)
  const [feedbackType, setFeedbackType] = useState("general")
  const [comments, setComments] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)

  const feedbackTypes = [
    { value: "accuracy", label: "Translation Accuracy" },
    { value: "clarity", label: "Audio Clarity" },
    { value: "timing", label: "Timing/Sync" },
    { value: "voice", label: "Voice Quality" },
    { value: "general", label: "General Feedback" },
  ]

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsSubmitting(true)

    const formData = new FormData()
    formData.append("rating", rating)
    formData.append("feedbackType", feedbackType)
    formData.append("comments", comments)
    formData.append("sourceLanguage", sourceLanguage || "")
    formData.append("targetLanguage", targetLanguage || "")
    formData.append("audioName", audioInput?.name || "")

    try {
      const res = await fetch(`${BASE_URL}/feedback`, {
        method: "POST",
        body: formData,
      })
      if (!res.ok) throw new Error("Server error")
      if (showToast) showToast("Thanks for your feedback!", "info")
      setRating(0)
      setFeedbackType("general")
      setComments("")
    } catch (err) {
      console.error("Feedback submit error:", err)
      if (showToast) showToast("Failed to submit feedback. Please try again.", "error")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="feedback-card">
      <h3>Send Feedback</h3>

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

        <button
          type="submit"
          className="submit-feedback-button"
          disabled={isSubmitting}
        >
          {isSubmitting ? "Submitting..." : "Submit Feedback"}
        </button>
      </form>
    </div>
  )
}
