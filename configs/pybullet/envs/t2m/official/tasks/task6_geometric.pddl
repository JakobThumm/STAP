(define (problem lh_lg_lp_1)
	(:domain geometric_workspace)
	(:objects
        rack - receptacle
        hook - tool
        red_box - box
		blue_box - box
        yellow_box - box
        cyan_box - box 
	)
	(:init
		(on rack table)
        (on hook table)
        (on red_box table)
		(on blue_box table)
        (on yellow_box table)
        (on cyan_box table)
        ; Geometric facts
        (inworkspace table)
        (inworkspace rack)
        (inworkspace hook)
        (inworkspace red_box)
        (inworkspace blue_box)
        (beyondworkspace yellow_box)        
        (beyondworkspace cyan_box)
	)
	(:goal (or
        (and
            (on red_box rack)
            (on blue_box rack)
            (on yellow_box rack)
            (on cyan_box table)
        )
        (and
            (on red_box rack)
            (on blue_box rack)
            (on yellow_box table)
            (on cyan_box rack)
        )
        (and
            (on red_box rack)
            (on blue_box table)
            (on yellow_box rack)
            (on cyan_box rack)
        )
        (and
            (on red_box table)
            (on blue_box rack)
            (on yellow_box rack)
            (on cyan_box rack)
        )
    ))
)
