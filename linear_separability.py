from manim import *

class LinearSeparability(Scene):
    def construct(self):
        # Coordinate plane
        plane = NumberPlane(
            x_range=[-0.5, 1.8, 1],
            y_range=[-0.5, 1.8, 1],
            x_length=7,
            y_length=7,
            background_line_style={
                "stroke_color": GREY,
                "stroke_opacity": 0.4,
            },
        )

        self.add(plane)

        # Axis labels
        x_label = Text("Input 1", font_size=28).next_to(
            plane.x_axis, DOWN
        )
        y_label = Text("Input 2", font_size=28).next_to(
            plane.y_axis, LEFT
        )

        self.add(x_label, y_label)

        # Points
        # Output 0: (0, 0)
        p0 = Dot(
            plane.c2p(0, 0),
            color=BLACK,
            radius=0.12
        )

        # Output 1: (1,0), (0,1), (1,1)
        p1 = Dot(plane.c2p(1, 0), color=BLUE, radius=0.12)
        p2 = Dot(plane.c2p(0, 1), color=BLUE, radius=0.12)
        p3 = Dot(plane.c2p(1, 1), color=BLUE, radius=0.12)

        # Labels
        label0 = Text("0", font_size=26, color=BLACK).next_to(p0, DOWN + LEFT)
        label1 = Text("1", font_size=26, color=BLUE).next_to(p1, DOWN)
        label2 = Text("1", font_size=26, color=BLUE).next_to(p2, LEFT)
        label3 = Text("1", font_size=26, color=BLUE).next_to(p3, UP + RIGHT)

        self.add(p0, p1, p2, p3)
        self.add(label0, label1, label2, label3)

        # Decision boundary:
        # x + y = 1
        #
        # To make sure the line doesn't pass through
        # any training point, we shift it slightly:
        # x + y = 0.8
        #
        # This places (0,0) on one side and all
        # three blue points on the other side.

        boundary = Line(
            plane.c2p(-0.3, 1.1),
            plane.c2p(1.1, -0.3),
            color=RED,
            stroke_width=6,
        )

        boundary_label = Text(
            "x₁ + x₂ = 0.8",
            color=RED,
            font_size=28,
        ).next_to(boundary, UP + RIGHT)

        self.add(boundary, boundary_label)

        # Explanation
        title = Text(
            "The classes are linearly separable",
            font_size=32
        ).to_edge(UP)

        self.play(Write(title))

        # Highlight the two classes
        blue_text = Text(
            "Blue = Output 1",
            color=BLUE,
            font_size=28
        ).to_edge(DOWN)

        black_text = Text(
            "Black = Output 0",
            color=BLACK,
            font_size=28
        ).next_to(blue_text, UP)

        self.play(
            Write(blue_text),
            Write(black_text)
        )

        self.wait(3)
