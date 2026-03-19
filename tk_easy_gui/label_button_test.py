import TkEasyGUI as eg

# ウィンドウの作成
layout = [
	[eg.Text("Hello, World!")],
	[eg.Button("OK")]
]
window = eg.Window("Hello", layout=layout)

# イベントループ
while window.is_alive():
	# イベントの取得
	event, values = window.read()
	# イベントの確認
	if event == "OK":
		eg.popup("Pushed OK Button")
		break
window.close()