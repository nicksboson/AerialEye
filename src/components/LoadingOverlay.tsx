import { motion, AnimatePresence } from 'framer-motion'
import { useEffect, useState } from 'react'
import { loadingMessages } from '../data/content'
import { Satellite, Check } from 'lucide-react'

interface LoadingOverlayProps {
  isVisible: boolean
  onComplete?: () => void
}

export default function LoadingOverlay({ isVisible, onComplete }: LoadingOverlayProps) {
  const [messageIndex, setMessageIndex] = useState(0)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!isVisible) {
      setMessageIndex(0)
      setProgress(0)
      return
    }

    const messageInterval = setInterval(() => {
      setMessageIndex((prev) => {
        if (prev >= loadingMessages.length - 1) {
          clearInterval(messageInterval)
          setTimeout(() => onComplete?.(), 500)
          return prev
        }
        return prev + 1
      })
    }, 350)

    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(progressInterval)
          return 100
        }
        return prev + 2
      })
    }, 50)

    return () => {
      clearInterval(messageInterval)
      clearInterval(progressInterval)
    }
  }, [isVisible, onComplete])

  const isComplete = messageIndex === loadingMessages.length - 1

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 backdrop-blur-xl"
        >
          {/* Subtle animated dots background */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            {Array.from({ length: 20 }).map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 rounded-full bg-teal-500/20"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                }}
                animate={{
                  scale: [1, 1.5, 1],
                  opacity: [0.2, 0.5, 0.2],
                }}
                transition={{
                  duration: 2 + Math.random() * 2,
                  repeat: Infinity,
                  delay: Math.random() * 2,
                  ease: 'easeInOut',
                }}
              />
            ))}
          </div>

          {/* Loading card */}
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="relative z-10 bg-white rounded-2xl p-6 sm:p-8 max-w-md w-full mx-4 shadow-large border border-stone-200"
          >
            {/* Animated icon */}
            <div className="flex justify-center mb-6">
              <motion.div
                animate={isComplete ? { scale: [1, 1.1, 1] } : { rotate: 360 }}
                transition={isComplete ? { duration: 0.3 } : { duration: 3, repeat: Infinity, ease: 'linear' }}
                className={`w-16 h-16 sm:w-20 sm:h-20 rounded-full flex items-center justify-center ${
                  isComplete ? 'bg-teal-100' : 'bg-teal-50'
                }`}
              >
                {isComplete ? (
                  <Check className="w-8 h-8 sm:w-10 sm:h-10 text-teal-600" />
                ) : (
                  <Satellite className="w-8 h-8 sm:w-10 sm:h-10 text-teal-600" />
                )}
              </motion.div>
            </div>

            {/* Message display */}
            <div className="h-6 sm:h-8 flex items-center justify-center mb-6">
              <AnimatePresence mode="wait">
                <motion.p
                  key={messageIndex}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className={`text-center font-medium text-sm sm:text-base ${
                    isComplete ? 'text-teal-700' : 'text-stone-700'
                  }`}
                >
                  {loadingMessages[messageIndex]}
                </motion.p>
              </AnimatePresence>
            </div>

            {/* Progress bar */}
            <div className="relative h-2 bg-stone-100 rounded-full overflow-hidden">
              <motion.div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-teal-500 to-teal-600 rounded-full"
                style={{ width: `${progress}%` }}
                transition={{ duration: 0.1 }}
              />
            </div>

            {/* Progress percentage */}
            <p className="text-center text-stone-500 text-xs sm:text-sm mt-3 font-medium">
              {Math.floor(progress)}%
            </p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
