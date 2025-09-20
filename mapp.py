# mental_health_mapping.py (or directly in app.py)
# This mapping provides structured responses for common mental health keywords.
# This helps the bot give direct, curated advice before falling back to the LLM's broader knowledge.
mental_health_mapping = {
    "anxiety": {
        "possible_concerns": ["Anxiety Disorder", "Generalized Anxiety", "Panic Attacks"],
        "advice": [
            "Try deep breathing exercises like the 4-7-8 technique.",
            "Practice grounding techniques, focusing on your five senses.",
            "Challenge anxious thoughts by asking if there's solid evidence for them.",
            "Limit caffeine and sugar intake, as they can worsen anxiety.",
            "Engage in light physical activity.",
            "Consider talking to a therapist or counselor for personalized strategies."
        ],
        "resources": [
            "Information on Cognitive Behavioral Therapy (CBT)",
            "Mindfulness for anxiety guides",
            "Local mental health services"
        ]
    },
    "depression": {
        "possible_concerns": ["Depression", "Low Mood", "Persistent Sadness"],
        "advice": [
            "Try to maintain a routine, including regular sleep and mealtimes.",
            "Engage in activities you once enjoyed, even if you don't feel like it at first.",
            "Connect with supportive friends or family members.",
            "Ensure you're getting some sunlight exposure if possible.",
            "Focus on small, achievable tasks to build a sense of accomplishment.",
            "A mental health professional can offer significant support through therapy or medication."
        ],
        "resources": [
            "Information on types of depression",
            "Support groups for depression",
            "Psychotherapy options"
        ]
    },
    "stress": {
        "possible_concerns": ["Chronic Stress", "Burnout", "Overwhelm"],
        "advice": [
            "Identify your stressors and try to manage or reduce them.",
            "Practice relaxation techniques like progressive muscle relaxation.",
            "Ensure you're getting adequate sleep.",
            "Incorporate regular physical activity into your day.",
            "Set realistic boundaries in your work and personal life.",
            "Consider delegating tasks if possible.",
            "Learning to say 'no' can be a powerful stress reducer."
        ],
        "resources": [
            "Stress management techniques",
            "Work-life balance tips",
            "Mindfulness for stress reduction"
        ]
    },
    "insomnia": {
        "possible_concerns": ["Sleep Disturbances", "Difficulty Falling/Staying Asleep"],
        "advice": [
            "Establish a consistent sleep schedule, even on weekends.",
            "Create a relaxing bedtime routine (e.g., warm bath, reading).",
            "Make sure your bedroom is dark, quiet, and cool.",
            "Avoid caffeine and heavy meals close to bedtime.",
            "Limit screen time an hour before bed.",
            "If worries keep you up, try writing them down earlier in the evening."
        ],
        "resources": [
            "Sleep hygiene guidelines",
            "Relaxation techniques for sleep",
            "Information on CBT for insomnia"
        ]
    },
    "sadness": { # Broader than depression, often a temporary emotion
        "possible_concerns": ["Low Mood", "Temporary Sadness"],
        "advice": [
            "Allow yourself to feel the emotion without judgment.",
            "Reach out to a friend or loved one to share how you're feeling.",
            "Engage in a comforting activity.",
            "Listen to uplifting music or watch a favorite movie.",
            "Practice self-compassion.",
            "If sadness persists or becomes overwhelming, seeking professional guidance is a good step."
        ],
        "resources": [
            "Emotional regulation strategies",
            "Self-compassion exercises"
        ]
    },
    "anger": {
        "possible_concerns": ["Anger Management Issues", "Irritability"],
        "advice": [
            "Take a deep breath and count to ten before reacting.",
            "Identify the triggers for your anger.",
            "Express your feelings assertively without aggression.",
            "Engage in physical activity to release tension.",
            "Practice empathy by trying to understand other perspectives.",
            "Therapy can provide effective strategies for managing anger."
        ],
        "resources": [
            "Anger management techniques",
            "Communication skills"
        ]
    },
    "loneliness": {
        "possible_concerns": ["Social Isolation", "Feelings of Loneliness"],
        "advice": [
            "Reach out to old friends or family members.",
            "Join a club or group with shared interests (online or in person).",
            "Volunteer for a cause you care about.",
            "Practice small acts of kindness to connect with others.",
            "Remember that many people experience loneliness, and it's okay to seek connection."
        ],
        "resources": [
            "Building social connections",
            "Community resources"
        ]
    },
    "grief": {
        "possible_concerns": ["Bereavement", "Loss"],
        "advice": [
            "Allow yourself to grieve, there's no 'right' way or timeline for it.",
            "Lean on your support system of friends and family.",
            "Remember to take care of your physical health during this time.",
            "Find healthy ways to express your emotions.",
            "Consider joining a bereavement support group.",
            "A therapist specializing in grief can offer guidance and support."
        ],
        "resources": [
            "Grief counseling options",
            "Understanding the stages of grief"
        ]
    },
    "overthinking": {
        "possible_concerns": ["Rumination", "Excessive Worry"],
        "advice": [
            "Set aside a 'worry time' each day to focus on concerns, then let them go.",
            "Engage in activities that require focus and distract you (e.g., puzzles, hobbies).",
            "Practice mindfulness to bring your attention to the present moment.",
            "Challenge negative thought patterns.",
            "Journaling can help you process and release thoughts.",
            "A mental health professional can teach you specific techniques to manage overthinking."
        ],
        "resources": [
            "Cognitive restructuring techniques",
            "Mindfulness exercises for overthinking"
        ]
    },
    "burnout": {
        "possible_concerns": ["Exhaustion", "Reduced Performance", "Cynicism"],
        "advice": [
            "Prioritize self-care and rest.",
            "Evaluate your workload and responsibilities; look for areas to delegate or reduce.",
            "Set clear boundaries between work and personal life.",
            "Reconnect with activities that bring you joy and relaxation.",
            "Talk to your supervisor or HR if work is a major factor.",
            "Therapy can help you develop strategies for prevention and recovery from burnout."
        ],
        "resources": [
            "Burnout prevention strategies",
            "Work-life balance resources"
        ]
    },
     "self-harm": {
        "possible_concerns": ["Self-Injurious Behavior"],
        "advice": [
            "It sounds like you're going through a lot. Please reach out for immediate help. You don't have to face this alone.",
            "Distraction techniques can sometimes help in the moment: hold ice, snap a rubber band on your wrist, draw on your skin with a red marker, or listen to loud music.",
            "Identify your triggers and try to avoid them or develop coping strategies for them.",
            "A mental health professional can provide strategies and support. This is not something you have to deal with on your own."
        ],
        "resources": [
            "Crisis Hotlines (e.g., 988 in the US)",
            "Urgent Mental Health Care",
            "Therapy specializing in self-harm"
        ]
    },
    "panic attack": {
        "possible_concerns": ["Panic Disorder"],
        "advice": [
            "Focus on your breath. Breathe in slowly through your nose for 4 counts, hold for 7, and exhale slowly through your mouth for 8.",
            "Remind yourself that this feeling will pass. It's intense, but not dangerous.",
            "Engage in grounding techniques: look around and name 5 things you can see, 4 things you can feel, 3 things you can hear, 2 things you can smell, and 1 thing you can taste.",
            "Splash cold water on your face or hold an ice pack.",
            "Move to a quiet, safe space.",
            "Learning coping mechanisms with a therapist can be very effective."
        ],
        "resources": [
            "Panic attack coping strategies",
            "Mindfulness for panic",
            "Therapy for panic disorder"
        ]
    }
    # Add more keywords and detailed advice as you gather more data
}