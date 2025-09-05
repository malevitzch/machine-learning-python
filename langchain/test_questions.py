
def strid(s):
    return s


basic_truth = (strid, [
    "What color is the sky?",
    "What is the value of 3 + 5?",
    "Can elephants fly?",
    "Is Corino a real country?"
])


def asktf(sentence):
    return f"Is the sentence \"{sentence}\" true?"


basic_false = (asktf, [
    "Elephants can fly.",
    "Corino is a real country."
])

basic_true = (asktf, [
    "Elephants can't fly.",
    "Corino isn't a real country."
])
