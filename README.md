# PROJECT - Probability Lab
# Monte Carlo simulation engine for discrete distributions



# Hey there!! 

Welcome to **Probability Lab** — a hands-on playground where math meets code.  
Instead of just staring at formulas, you can **see probability distributions come alive**, tweak parameters, and watch how randomness behaves. Perfect for students, hobbyists, or anyone curious about probability.  

---

# What it does ???

- **Interactive sliders**: Control probability, number of trials, mean (μ), standard deviation (σ), and sample size.  
- **Animations**: Watch simulations unfold in real-time — like seeing the law of large numbers in action!  
- **Theoretical curves**: Compare your simulations with what theory predicts.  
- **Theory insights**: Each distribution comes with a short explanation, right next to the graph.  
- Dark-themed and fully responsive, so your eyes don’t hurt while exploring randomness.  

---

##  Distributions you can play with. (visualizing distributions) 

1. **Binomial** – Count the number of successes in repeated trials.  
2. **Geometric** – How long until your first success?  
3. **Poisson** – Model rare events like accidents or arrivals.  
4. **Normal** – Central Limit Theorem magic.  
5. **Exponential** – Time between random events.  

## How to try it

Clone the repo, set up a venv , install dependencies, and run the app it's simple: try it out..

```bash
git clone https://github.com/Deshan-5/probability-lab.git
cd probability-lab
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run streamlit_app.py

Then watch the magic happen in your browser! 

Try it online:
Deployed on Streamlit Community Cloud.



Built with:

Python 3.13
Streamlit
NumPy
Matplotlib