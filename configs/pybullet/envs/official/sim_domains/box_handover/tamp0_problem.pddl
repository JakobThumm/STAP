(define (problem hook-handover-0)
	(:domain workspace)
	(:objects
		hook - movable
		red_box - movable
        screwdriver - movable
	)
	(:init
		(on hook table)
        (on screwdriver table)
		(on red_box table)
	)
	(:goal (and
		(inhand hook)
	))
)
