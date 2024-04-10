(define (problem screwdriver-pick-0)
	(:domain workspace)
	(:objects
		red_box - movable
        screwdriver - movable
	)
	(:init
		(ingripper red_box)
        (on screwdriver table)
	)
	(:goal (and
		(on red_box table)
	))
)
