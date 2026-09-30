# Caption validation: caption_val_v3.jsonl (50 ViPE videos)

| stat | Qwen recaption | original HF `long` |
|---|---|---|
| n | 50 | 50 |
| words_mean | 116.280 | 122.660 |
| words_median | 115.500 | 123.500 |
| words_p5 | 89.350 | 95.650 |
| words_p95 | 146.750 | 146.000 |
| words_min | 83 | 27 |
| words_max | 169 | 153 |
| sentences_mean | 6.520 | 5.700 |
| starts_with_a_an | 0.760 | 0.740 |
| starts_with_the_video_or_in_this | 0.100 | 0.040 |
| has_newline_or_list | 0.000 | 0.000 |
| mentions_camera | 0.220 | 0.500 |
| top_opening_share | 0.040 | 0.040 |
| starts_young_woman_long_dark_hair | 0.000 | 0.000 |

Embedding cosine (all-mpnet-base-v2): matched mean 0.861 (min 0.768), mismatched mean 0.322; retrieval top-1 (Qwen -> original) 1.00

## Examples

**00019757** (124 vs 145 words)

- Qwen: Two children sit at a small wooden table in a kitchen, engaged in eating. The boy on the left wears a dark green long-sleeved shirt and blue jeans, while the girl on the right dons a white t-shirt and striped pants. Both have their hair tied back, the boy’s in a ponytail and the girl’s in two braids. They use utensils to eat from black plates placed before them. A green glass sits between them on the table. The kitchen has light-colored walls, a window above the table allowing natural light to stream in, and a tiled backsplash with pastel hues of pink, blue, and beige. The children appear focused on their meal, occasionally looking at each other, creating an atmosphere of casual domesticity.
- orig: Two children, a boy and a girl, are seated across from each other at a rustic wooden dining table in a cozy kitchen. The boy, with shoulder-length brown hair and wearing a dark green long-sleeved shirt and jeans, is smiling and laughing as he eats from his black plate using a fork. The girl, with blonde hair tied into a high ponytail secured with a green hair tie and dressed in a white t-shirt and striped pants, is also eating with a spoon, her expression animated as she enjoys her meal. Between them on the table sits a translucent green glass. The kitchen features light-colored cabinets and a patchwork tile backsplash, with sunlight streaming through a window on the right, illuminating the scene. Both children are engaged in conversation, their eyes locked on each other, sharing laughter and the simple joy of a shared meal.

**00021511** (124 vs 117 words)

- Qwen: A woman with long brown hair tied back in a ponytail sits on the floor of a well-lit gym, wearing a black sports bra and black leggings. She holds a small blue container close to her mouth, taking a bite from it while maintaining eye contact with the camera. Her right leg is bent at the knee, resting flat on the mat, while her left leg is extended outwards. The background features various gym equipment, including TRX suspension straps hanging from the ceiling and colorful rings mounted on the wall. The setting is clean and organized, with bright lighting illuminating the space evenly. The woman appears focused and engaged, suggesting she might be demonstrating or explaining something related to the contents of the container.
- orig: A fit woman with defined muscles sits cross-legged on a black exercise mat in a well-equipped gym, dressed in a black sports bra and leggings. She holds a small blue packet in her right hand, bringing it to her lips and taking a bite, her eyes closing briefly as she savors the flavor. Her left hand rests gently on her thigh, while her head tilts slightly as she chews. The gym background features hanging resistance bands, kettlebells, and gymnastic rings, with bright lighting illuminating the space. Her expression shifts from focused to satisfied as she enjoys her snack, occasionally glancing around the room with a smile. The atmosphere conveys a sense of energy and dedication to fitness.

**00002010** (123 vs 142 words)

- Qwen: A man stands confidently in a grand hall adorned with intricate carvings and warm, golden hues. His attire is elaborate, featuring a rich brown robe with ornate gold embellishments and a matching crown perched atop his head. The man’s hands are clasped together in front of him, suggesting a moment of contemplation or authority. The room is filled with tall, ornate columns and numerous torches casting flickering shadows across the stone walls, adding to the ambiance of the space. The ceiling above him features an intricate, circular design with light streaming through, creating a serene yet imposing atmosphere. The overall impression is one of grandeur and historical significance, as if the man is a figure of importance within a palace or ancient temple.
- orig: A regal figure stands at the center of an opulent, ancient hall, bathed in dramatic beams of light piercing through an ornate, circular ceiling design. Dressed in an intricately detailed bronze and brown ceremonial armor, complete with shoulder guards, arm bracers, and a flowing cape, the man exudes authority and solemnity. His hands are clasped in front of him, and his gaze is fixed forward, suggesting he is addressing an unseen audience or preparing for a momentous decision. The hall is flanked by towering pillars adorned with golden embellishments, and numerous braziers with flickering flames cast a warm, golden glow across the room. Smoke drifts gently in the air, adding to the solemn and majestic atmosphere. The setting evokes a sense of power, tradition, and divine presence, as if the man is a ruler or a high priest in a sacred ritual.

**00013240** (105 vs 137 words)

- Qwen: A bride sits on a red leather couch, her head adorned with a white veil decorated with pearls and a matching tiara. She wears a strapless wedding gown with intricate detailing and a statement necklace. Her hair is styled in an elegant updo, and she has long, manicured nails painted in a light color. She is clasping her hands together, looking down. Another woman, dressed in black with long, wavy hair, approaches from behind, adjusting the bride’s veil. The background is softly lit, highlighting the red couch and the bride's attire. The setting appears intimate and focused on the bride's preparation moments before the ceremony.
- orig: A bride, adorned in a sparkling white wedding gown, sits gracefully on a rich red leather couch, her hair elegantly styled in an updo with a delicate floral headpiece and veil. She wears a dazzling diamond necklace and matching earrings, her nails painted a soft white. Her hands are clasped together, and she gazes downward with a gentle smile, suggesting a moment of quiet reflection or anticipation. As the camera slowly pans to the right, another woman in a black dress with sheer sleeves and a bow tie enters the frame, leaning in close to the bride. She has long, wavy hair and bold makeup, including dark eyeliner and red lipstick. She gently places her hand over the bride’s, offering comfort and support, while both women share a tender, intimate moment, their expressions conveying warmth and affection.

**00026165** (118 vs 123 words)

- Qwen: A cyclist wearing a gray long-sleeve shirt, blue shorts, and black gloves rides a light blue bicycle on a paved road. A bright yellow saddle bag is strapped securely to the bike frame. The cyclist's face is bearded, and he wears glasses. He pedals steadily forward, his arms positioned comfortably at the handlebars. The road stretches ahead, flanked by a stone barrier on the right and a steep, rugged mountain slope on the left. The sun shines brightly overhead, casting a warm, golden glow across the scene. The sky is clear and blue, suggesting a crisp, sunny day. The overall mood is one of adventure and tranquility, as the cyclist enjoys the open road and stunning natural surroundings.
- orig: A bearded man with glasses is cycling along a winding mountain road under a bright, clear sky. He wears a gray long-sleeve shirt and blue shorts, gripping the handlebars of a light blue bicycle that has a yellow bag strapped across the front. The sun shines intensely from behind a towering rocky cliff, casting radiant beams and lens flares across the scene. To his right, a stone barrier lines the road, while the left side features a steep, rocky slope with sparse vegetation. In the distance, the valley unfolds with rolling hills and more mountain peaks under a vast blue sky. The man pedals steadily, maintaining a steady gaze forward, embodying the spirit of adventure and endurance as he navigates the scenic route.

