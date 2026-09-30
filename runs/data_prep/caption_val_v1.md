# Caption validation: caption_val_v1.jsonl (50 ViPE videos)

| stat | Qwen recaption | original HF `long` |
|---|---|---|
| n | 50 | 50 |
| words_mean | 127.800 | 122.660 |
| words_median | 127.000 | 123.500 |
| words_p5 | 90.450 | 95.650 |
| words_p95 | 153.200 | 146.000 |
| words_min | 82 | 27 |
| words_max | 184 | 153 |
| sentences_mean | 6.820 | 5.700 |
| starts_with_a_an | 0.900 | 0.740 |
| starts_with_the_video_or_in_this | 0.060 | 0.040 |
| has_newline_or_list | 0.000 | 0.000 |
| mentions_camera | 0.820 | 0.500 |

Embedding cosine (all-mpnet-base-v2): matched mean 0.846 (min 0.626), mismatched mean 0.355; retrieval top-1 (Qwen -> original) 0.94

## Examples

**00019757** (142 vs 145 words)

- Qwen: A young boy and a little girl sit opposite each other at a small wooden table, both dressed casually. The boy wears a dark green long-sleeved shirt and blue jeans, while the girl sports a white t-shirt and a striped skirt. They are engaged in eating, using spoons to scoop food from black plates set on brown paper placemats. The boy holds his spoon mid-movement, while the girl brings hers closer to her mouth. A green glass sits between them on the table. Sunlight streams through a window behind them, casting warm light across the scene. The background includes a tiled backsplash and a white cabinet, adding to the homely feel. The camera remains static throughout, capturing the intimate moment of the children enjoying their meal together. The overall mood is serene and domestic, highlighting a quiet, everyday moment shared between siblings.
- orig: Two children, a boy and a girl, are seated across from each other at a rustic wooden dining table in a cozy kitchen. The boy, with shoulder-length brown hair and wearing a dark green long-sleeved shirt and jeans, is smiling and laughing as he eats from his black plate using a fork. The girl, with blonde hair tied into a high ponytail secured with a green hair tie and dressed in a white t-shirt and striped pants, is also eating with a spoon, her expression animated as she enjoys her meal. Between them on the table sits a translucent green glass. The kitchen features light-colored cabinets and a patchwork tile backsplash, with sunlight streaming through a window on the right, illuminating the scene. Both children are engaged in conversation, their eyes locked on each other, sharing laughter and the simple joy of a shared meal.

**00021511** (148 vs 117 words)

- Qwen: A young woman with long dark hair tied back into a ponytail is seated on a yoga mat in a gym, wearing a black sports bra and black leggings. She holds a small container of what appears to be protein powder or a similar supplement in her right hand while her left hand rests on her bent knee. She brings the container to her mouth and takes a sip, then sets it down. The background reveals various gym equipment, including TRX straps hanging from the ceiling and kettlebells lined up against the wall. The setting is brightly lit, likely by overhead lights, which cast a clear view of the woman and the surrounding environment. The camera remains stationary throughout the clip, focusing steadily on the woman as she completes her action. The overall mood of the scene is focused and active, reflecting the context of a workout session.
- orig: A fit woman with defined muscles sits cross-legged on a black exercise mat in a well-equipped gym, dressed in a black sports bra and leggings. She holds a small blue packet in her right hand, bringing it to her lips and taking a bite, her eyes closing briefly as she savors the flavor. Her left hand rests gently on her thigh, while her head tilts slightly as she chews. The gym background features hanging resistance bands, kettlebells, and gymnastic rings, with bright lighting illuminating the space. Her expression shifts from focused to satisfied as she enjoys her snack, occasionally glancing around the room with a smile. The atmosphere conveys a sense of energy and dedication to fitness.

**00002010** (128 vs 142 words)

- Qwen: A man stands confidently in a grand hall adorned with intricate carvings and warm, golden light filtering through ornate windows. He wears an elaborate outfit featuring a rich brown tunic and gold armor with decorative patterns, complemented by a flowing red cape draped over his shoulders. His hands are clasped together in front of him, and he appears to be addressing an unseen audience. The setting is opulent, with tall columns supporting the ceiling and multiple torches casting flickering shadows around the room. The camera remains steady, focusing on the man as he maintains his posture, suggesting a moment of importance or declaration within this regal space. The overall atmosphere is one of solemnity and grandeur, enhanced by the soft, ambient lighting and the detailed architecture surrounding him.
- orig: A regal figure stands at the center of an opulent, ancient hall, bathed in dramatic beams of light piercing through an ornate, circular ceiling design. Dressed in an intricately detailed bronze and brown ceremonial armor, complete with shoulder guards, arm bracers, and a flowing cape, the man exudes authority and solemnity. His hands are clasped in front of him, and his gaze is fixed forward, suggesting he is addressing an unseen audience or preparing for a momentous decision. The hall is flanked by towering pillars adorned with golden embellishments, and numerous braziers with flickering flames cast a warm, golden glow across the room. Smoke drifts gently in the air, adding to the solemn and majestic atmosphere. The setting evokes a sense of power, tradition, and divine presence, as if the man is a ruler or a high priest in a sacred ritual.

**00013240** (150 vs 137 words)

- Qwen: A young woman with long dark hair styled in loose curls wears a black dress with sheer sleeves and a red lipstick. She approaches another woman who is seated on a red leather couch. The seated woman has her hair elegantly styled into an updo adorned with a veil and jewelry, including a large necklace and earrings. She is dressed in a white bridal gown. The standing woman gently adjusts the seated woman’s veil, smoothing it down and tucking it behind her ear. The setting appears to be indoors, possibly a hotel room or a private residence, with warm lighting that highlights the rich red of the couch and the sparkle of the jewelry. The camera remains stationary throughout the sequence, focusing closely on the interaction between the two women. The overall mood is intimate and serene, capturing a moment of preparation before what seems to be a significant event.
- orig: A bride, adorned in a sparkling white wedding gown, sits gracefully on a rich red leather couch, her hair elegantly styled in an updo with a delicate floral headpiece and veil. She wears a dazzling diamond necklace and matching earrings, her nails painted a soft white. Her hands are clasped together, and she gazes downward with a gentle smile, suggesting a moment of quiet reflection or anticipation. As the camera slowly pans to the right, another woman in a black dress with sheer sleeves and a bow tie enters the frame, leaning in close to the bride. She has long, wavy hair and bold makeup, including dark eyeliner and red lipstick. She gently places her hand over the bride’s, offering comfort and support, while both women share a tender, intimate moment, their expressions conveying warmth and affection.

**00026165** (127 vs 123 words)

- Qwen: A bearded man wearing glasses, a light gray long-sleeve shirt, blue shorts, and black gloves rides a light blue road bike along a winding mountain road. He carries a bright yellow pannier bag on the bike. The road curves gently, bordered by a stone barrier on the right and a steep, rocky hillside on the left. The sun shines brightly in the sky, casting a warm glow and creating lens flares that dance across the frame. The landscape stretches out into the distance, revealing more rugged mountains under a clear blue sky. The camera follows the cyclist from behind, maintaining a steady pace as he pedals forward. The overall mood is one of adventure and tranquility, capturing the essence of a solitary journey through stunning natural beauty.
- orig: A bearded man with glasses is cycling along a winding mountain road under a bright, clear sky. He wears a gray long-sleeve shirt and blue shorts, gripping the handlebars of a light blue bicycle that has a yellow bag strapped across the front. The sun shines intensely from behind a towering rocky cliff, casting radiant beams and lens flares across the scene. To his right, a stone barrier lines the road, while the left side features a steep, rocky slope with sparse vegetation. In the distance, the valley unfolds with rolling hills and more mountain peaks under a vast blue sky. The man pedals steadily, maintaining a steady gaze forward, embodying the spirit of adventure and endurance as he navigates the scenic route.

