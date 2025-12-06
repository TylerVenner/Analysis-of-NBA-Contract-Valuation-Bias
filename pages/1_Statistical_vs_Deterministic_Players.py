import streamlit as st
import pandas as pd
from configs.vis_utils import load_data




st.markdown("""
##  Statistical vs. Deterministic Players

Not every NBA salary is created the same way.

Some players are paid by **pure formula** – their contracts are locked in by league rules, not by what a team is willing to offer in an open negotiation. These are the **deterministic players**: rookies on the fixed Rookie Scale, stars on pre-defined Max deals, and veterans on minimum contracts. Their salaries are mostly decided before a single shot is taken.

Other players live in the **statistical world**. Their contracts are negotiated in the open market, where performance, timing, leverage, and perception all collide. These are the players our models can truly “listen to” when learning how the NBA values different skills and attributes.

---

##  What Our Salary Data Lets Us See

Our dataset combines:
- **Player salary** (how much they actually get paid),
- **On-court performance stats** (points, shooting, defense, impact metrics),
- **Contextual factors** like age, draft position, team market, and popularity signals.

By separating **deterministic contracts** from **free-market contracts**, we can:
- Learn how the market prices different skills *only* from players whose salaries are truly negotiable.
- Then **apply** those learned prices to everyone else, including rookies and max players, to see where the system itself creates distortions.

In other words, we ask:
> “If this rookie or max-contract player were paid like a normal free-market player, what would their salary look like?”

That counterfactual lets us see where the rules of the system, not just performance, are driving big gaps.

---

##  A Simple Question: Wemby vs. Curry

Take a thought experiment:

- **Victor Wembanyama**: a young player who puts up elite defensive stats, blocks a ton of shots, and warps the floor on defense.
- **Stephen Curry**: a veteran whose block numbers are tiny, but whose shooting, gravity, and global popularity are off the charts.

On paper, you might ask:
> “If Wemby has way more blocks, rim protection, and defensive impact, why does Curry get paid so much more?”

Our project doesn’t just shrug and say “because he’s a star.”  
We try to break that down into **measurable pieces**:

- Which **stats actually move salary** once you control for everything else?
- How much of a player’s paycheck is tied to **age**, **experience**, or **draft history**, instead of pure performance?
- How big is the **popularity premium** — things like social media following, brand power, and being the face of a franchise?
- And for rookies like Wemby, how much is their value **held back** by deterministic contract rules, even when their performance already looks elite?

---

## What You’ll See on This Page

Below, we highlight preliminary visualizations that start to answer these questions, such as:

- **Salary vs. Performance Scatterplots**  
  Showing how free-market players align (or don’t) with their production.

- **Deterministic vs. Statistical Groups**  
  Comparing rookies, max players, and open-market veterans to see where the biggest gaps emerge.

- **Case Study Points**  
  Marking players like Wembanyama and Curry on these plots to visualize just how differently the system treats them.

These visuals are the first step in moving from “Curry just gets paid more” to a clearer, data-driven story:  
**Which attributes the NBA really rewards, how the rules distort that value, and who ends up caught in the middle.**
""")

