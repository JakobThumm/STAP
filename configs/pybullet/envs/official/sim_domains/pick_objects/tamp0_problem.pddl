(define (problem screwdriver-pick-0)
	(:domain workspace)
	(:objects
		red_box - movable
        screwdriver - movable
	)
	(:init
		(on red_box table)
        (on screwdriver table)
	)
	(:goal (and
		(ingripper screwdriver)
	))
)
