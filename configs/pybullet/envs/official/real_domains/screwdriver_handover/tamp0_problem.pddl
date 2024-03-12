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
        (on screwdriver table)
	)
	(:goal (and
		(inhand screwdriver)
	))
)
