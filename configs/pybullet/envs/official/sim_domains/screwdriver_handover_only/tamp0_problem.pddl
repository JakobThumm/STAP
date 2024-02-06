(define (problem screwdriver-handover-0)
	(:domain workspace)
	(:objects
		hook - movable
		red_box - movable
        screwdriver - movable
	)
	(:init
		(on hook table)
		(on red_box table)
        (ingripper screwdriver)
	)
	(:goal (and
		(inhand screwdriver)
	))
)
