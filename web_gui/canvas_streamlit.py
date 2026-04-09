import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.title("Drawable Canvas")

# キャンバスの作成
canvas_result = st_canvas(
	fill_color="rgba(255, 165, 0, 0.3)",
	stroke_width=2,
	stroke_color="#000000",
	background_color="#EEEEEE",
	update_streamlit=True, # マウスアップ時にデータを送信
	height=300,
	drawing_mode="point",
	key="canvas",
)

if canvas_result.json_data is not None:
	# 描画データ（クリックした座標など）を取得
	st.write(canvas_result.json_data)
