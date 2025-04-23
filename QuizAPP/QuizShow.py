import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import random

# ========== DATA ==========
quiz_data = [
    {
        "question": "What is the capital of France?",
        "options": ["Paris", "London", "Berlin", "Madrid"],
        "answer": "Paris",
        "image": "paris.jpg"
    },
    {
        "question": "What is 2 + 2?",
        "options": ["3", "4", "5", "6"],
        "answer": "4",
        "image": "math.jpg"
    },
    {
        "question": "Who wrote 'Hamlet'?",
        "options": ["Charles Dickens", "Leo Tolstoy", "William Shakespeare", "Jane Austen"],
        "answer": "William Shakespeare",
        "image": "shakespeare.jpg"
    },
    {
        "question": "What planet is known as the Red Planet?",
        "options": ["Earth", "Venus", "Mars", "Jupiter"],
        "answer": "Mars",
        "image": "mars.jpg"
    },
    {
        "question": "Python is a snake.",
        "options": ["✅", "❌"],
        "answer": "❌"
    },
    {
        "question": "The Pacific Ocean is the largest ocean.",
        "options": ["✅", "❌"],
        "answer": "✅"
    },
    {
        "question": "The square root of 16 is 4.",
        "options": ["✅", "❌"],
        "answer": "✅"
    },
    {
        "question": "The sun is a planet.",
        "options": ["✅", "❌"],
        "answer": "❌"
    },
    {
        "question": "The currency of Japan is Yen.",
        "options": ["✅", "❌"],
        "answer": "✅"
    },
    {
        "question": "Water boils at 90°C.",
        "options": ["✅", "❌"],
        "answer": "❌"
    },
    {
        "question": "What is the chemical symbol for water?",
        "options": ["O2", "H2", "CO2", "H₂O"],
        "answer": "H₂O"
    },
    {
        "question": "How many continents are there?",
        "options": ["5", "6", "7", "8"],
        "answer": "7"
    },
    {
        "question": "Which gas do plants absorb?",
        "options": ["Oxygen", "Hydrogen", "Carbon Dioxide", "Nitrogen"],
        "answer": "Carbon Dioxide"
    },
    {
        "question": "Which language is primarily spoken in Brazil?",
        "options": ["Spanish", "English", "Portuguese", "French"],
        "answer": "Portuguese"
    },
    {
        "question": "Which organ pumps blood?",
        "options": ["Lungs", "Brain", "Heart", "Liver"],
        "answer": "Heart"
    }
]

random.shuffle(quiz_data)

# ========== APP SETUP ==========
root = tk.Tk()
root.title("Quiz App")

question_index = 0
score = 0
wrong_answers = []

# ========== FUNCTIONS ==========

def load_question():
    global current_image
    q = quiz_data[question_index]
    question_label.config(text=q["question"])

    for i in range(4):
        if i < len(q["options"]):
            option_buttons[i].config(text=q["options"][i], state=tk.NORMAL)
            option_buttons[i].pack()
        else:
            option_buttons[i].pack_forget()

    if "image" in q:
        try:
            img = Image.open(q["image"])
            img = img.resize((200, 150))
            current_image = ImageTk.PhotoImage(img)
            image_label.config(image=current_image)
        except:
            image_label.config(image='')
    else:
        image_label.config(image='')

def check_answer(selected):
    global question_index, score
    q = quiz_data[question_index]
    answer = q["answer"]
    if selected == answer:
        score += 1
    else:
        wrong_answers.append((q["question"], selected, answer))

    question_index += 1
    if question_index < len(quiz_data):
        load_question()
    else:
        show_result()

def show_result():
    result = f"You got {score}/{len(quiz_data)} correct."
    if wrong_answers:
        result += "\n\nIncorrect Answers:\n"
        for q, s, a in wrong_answers:
            result += f"- {q}\n  Your answer: {s}\n  Correct answer: {a}\n\n"
    messagebox.showinfo("Quiz Result", result)
    root.quit()

# ========== WIDGETS ==========

question_label = tk.Label(root, text="", wraplength=400, font=("Arial", 16), justify="center")
question_label.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

option_buttons = []
for i in range(4):
    btn = tk.Button(root, text="", width=25, font=("Arial", 14), command=lambda i=i: check_answer(option_buttons[i]["text"]))
    btn.pack(pady=5)
    option_buttons.append(btn)

# ========== START QUIZ ==========
load_question()
root.mainloop()
