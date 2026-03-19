import flet as ft

def main(page: ft.Page):
	page.title = "Mouse Event Coordinates"
	page.vertical_alignment = ft.MainAxisAlignment.CENTER
	page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

	coords_text = ft.Text("Move your mouse over the box", size=16)

	def on_hover_event(e: ft.HoverEvent):
		"""Handler for the mouse hover event."""
		coords_text.value = (
			f"Global: x={e.global_position.x:.2f}, y={e.global_position.y:.2f} | "
			f"Local: x={e.local_position.x:.2f}, y={e.local_position.y:.2f}"
		)
		page.update()

	# The GestureDetector can wrap any control to capture its events
	interactive_area = ft.GestureDetector(
		content=ft.Container(
			width=200,
			height=200,
			bgcolor=ft.Colors.BLUE_GREY_100,
			border_radius=ft.border_radius.all(10),
			alignment=ft.MainAxisAlignment.CENTER,
			content=ft.Text("Hover here"),
		),
		on_hover=on_hover_event,
		hover_interval=10 # Throttle events to every 10 milliseconds
	)

	page.add(
		interactive_area,
		coords_text
	)

ft.app(target=main)
