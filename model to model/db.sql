CREATE TABLE faqs (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL
);
INSERT INTO faqs (question, answer) VALUES
("What is your return policy?", "Our return policy is..."),
("How do I track my order?", "You can track your order...");
