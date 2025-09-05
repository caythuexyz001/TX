import random
from collections import Counter

class MarkovPredictor:
    def __init__(self):
        self.history = []  # danh sách kết quả T / X

    def update(self, result: str):
        """Cập nhật kết quả thực tế"""
        if result not in ["T", "X"]:
            return
        self.history.append(result)

    def predict(self):
        """Dự đoán ván kế tiếp"""
        if len(self.history) < 3:
            return random.choice(["T", "X"]), "Random"

        pattern = tuple(self.history[-3:])
        candidates = [
            self.history[i+3]
            for i in range(len(self.history)-3)
            if tuple(self.history[i:i+3]) == pattern
        ]
        if candidates:
            most_common = Counter(candidates).most_common(1)[0][0]
            return most_common, f"Markov: {''.join(pattern)} → {most_common}"
        return random.choice(["T", "X"]), "Pattern not found"

    def stats(self):
        total = len(self.history)
        t = self.history.count("T")
        x = self.history.count("X")
        return {
            "total": total,
            "tai": t,
            "xiu": x,
            "p_tai": (t/total*100) if total else 0,
            "p_xiu": (x/total*100) if total else 0,
        }

    def streaks(self):
        """Phân tích chuỗi tay ăn / gãy"""
        if not self.history:
            return []
        streaks = []
        cur = self.history[0]
        count = 1
        for h in self.history[1:]:
            if h == cur:
                count += 1
            else:
                streaks.append((cur, count))
                cur = h
                count = 1
        streaks.append((cur, count))
        return streaks[-5:]  # trả về 5 chuỗi gần nhất
