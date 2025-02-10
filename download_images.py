from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

keywords = ["seizure spectograph", "normal spectograph", "pre-ictal spectograph"] #pre-ictal is questionable

for kw in keywords:
    response().download(kw,5)