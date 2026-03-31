MODULE HELPERS
    USE NETCDF
    IMPLICIT NONE
    PRIVATE

    INTEGER, PARAMETER, PUBLIC :: DP = KIND(1.0D0)
    PUBLIC :: NC_CHECK, ERFINV, OVERLAP_1D, DEG_TO_M_DX, DEG_TO_M_DY, RECT_SLICE_RAW_OVERLAP
    PUBLIC :: RECT_SLICE_RAW_OVERLAP_FACE

    ! ---------- GENERIC NETCDF ERROR CHECKING ----------
CONTAINS
    SUBROUTINE NC_CHECK(STATUS, WHERE)
        INTEGER, INTENT(IN) :: STATUS
        CHARACTER(LEN=*), INTENT(IN) :: WHERE
        IF (STATUS /= NF90_NOERR) THEN
        PRINT *, "NETCDF ERROR IN ", TRIM(WHERE), ": ", TRIM(NF90_STRERROR(STATUS))
        STOP 1
        END IF
    END SUBROUTINE NC_CHECK

    PURE FUNCTION ERFINV(X) RESULT(Y)
        ! Approximate inverse error function for |x| < 1
        REAL(DP), INTENT(IN) :: X
        REAL(DP) :: Y
        REAL(DP) :: A, LN1
        REAL(DP) :: SX
        REAL(DP), PARAMETER :: PI = 3.14159265358979323846_DP

        ! Winitzki approximation
        A = 0.147_DP
        SX = SIGN(1.0_DP, X)
        LN1 = LOG(1.0_DP - X*X)
        Y = SX * SQRT( SQRT( (2.0_DP/(PI*A) + 0.5_DP*LN1)**2 - LN1/A ) - (2.0_DP/(PI*A) + 0.5_DP*LN1) )
    END FUNCTION ERFINV

    PURE FUNCTION OVERLAP_1D(A_MIN, A_MAX, B_MIN, B_MAX) RESULT(OL)
        REAL(DP), INTENT(IN) :: A_MIN, A_MAX, B_MIN, B_MAX
        REAL(DP) :: OL

        OL = MAX(0.0_DP, MIN(A_MAX, B_MAX) - MAX(A_MIN, B_MIN))
    END FUNCTION OVERLAP_1D

    PURE FUNCTION DEG_TO_M_DX(DLON_DEG, LAT_DEG) RESULT(DX_M)
        REAL(DP), INTENT(IN) :: DLON_DEG, LAT_DEG
        REAL(DP) :: DX_M
        REAL(DP), PARAMETER :: PI = 3.14159265358979323846_DP
        REAL(DP), PARAMETER :: EARTH_RADIUS_M = 6371000.0_DP

        DX_M = EARTH_RADIUS_M * COS(LAT_DEG * PI / 180.0_DP) * DLON_DEG * PI / 180.0_DP
    END FUNCTION DEG_TO_M_DX

    PURE FUNCTION DEG_TO_M_DY(DLAT_DEG) RESULT(DY_M)
        REAL(DP), INTENT(IN) :: DLAT_DEG
        REAL(DP) :: DY_M
        REAL(DP), PARAMETER :: PI = 3.14159265358979323846_DP
        REAL(DP), PARAMETER :: EARTH_RADIUS_M = 6371000.0_DP

        DY_M = EARTH_RADIUS_M * DLAT_DEG * PI / 180.0_DP
    END FUNCTION DEG_TO_M_DY

    FUNCTION RECT_SLICE_RAW_OVERLAP(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, SLICE_BOX) RESULT(RAW)
        IMPLICIT NONE
        REAL(DP), INTENT(IN) :: X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX
        REAL(DP), INTENT(IN) :: SLICE_BOX(8, 3)
        REAL(DP) :: RAW
        REAL(DP), PARAMETER :: EPS = 1.0E-12_DP
        INTEGER :: NFACE
        REAL(DP) :: RAW_FACE, RAW_V
        REAL(DP) :: Z_SLICE_MIN, Z_SLICE_MAX
        REAL(DP) :: FOOTPRINT(32,2)

        RAW = 0.0_DP
        NFACE = SIZE(SLICE_BOX, 1)

        IF (NFACE == 4) THEN
            RAW = RECT_SLICE_RAW_OVERLAP_FACE(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, SLICE_BOX)

        ELSE IF (NFACE == 8) THEN
            FOOTPRINT(:,:) = 0.0_DP

            ! back-left  = mean(BL, TL)
            FOOTPRINT(1,1) = 0.5_DP * (SLICE_BOX(1,1) + SLICE_BOX(2,1))
            FOOTPRINT(1,2) = 0.5_DP * (SLICE_BOX(1,2) + SLICE_BOX(2,2))

            ! front-left = mean(BL, TL) on next face
            FOOTPRINT(2,1) = 0.5_DP * (SLICE_BOX(5,1) + SLICE_BOX(6,1))
            FOOTPRINT(2,2) = 0.5_DP * (SLICE_BOX(5,2) + SLICE_BOX(6,2))

            ! front-right = mean(TR, BR) on next face
            FOOTPRINT(3,1) = 0.5_DP * (SLICE_BOX(7,1) + SLICE_BOX(8,1))
            FOOTPRINT(3,2) = 0.5_DP * (SLICE_BOX(7,2) + SLICE_BOX(8,2))

            ! back-right = mean(TR, BR)
            FOOTPRINT(4,1) = 0.5_DP * (SLICE_BOX(3,1) + SLICE_BOX(4,1))
            FOOTPRINT(4,2) = 0.5_DP * (SLICE_BOX(3,2) + SLICE_BOX(4,2))

            RAW_FACE = POLYRECT_INTERSECTION_AREA(FOOTPRINT, 4, X_MIN, X_MAX, Y_MIN, Y_MAX)

            Z_SLICE_MIN = MINVAL(SLICE_BOX(:,3))
            Z_SLICE_MAX = MAXVAL(SLICE_BOX(:,3))
            RAW_V = OVERLAP_1D(Z_MIN, Z_MAX, Z_SLICE_MIN, Z_SLICE_MAX)

            IF (RAW_FACE > EPS .AND. RAW_V > EPS) THEN
                RAW = RAW_FACE * RAW_V
            END IF
        END IF
    END FUNCTION RECT_SLICE_RAW_OVERLAP

    FUNCTION POLYRECT_INTERSECTION_AREA(POLY, N, X_MIN, X_MAX, Y_MIN, Y_MAX) RESULT(AREA)
        IMPLICIT NONE
        REAL(DP), INTENT(IN) :: POLY(32,2)
        INTEGER, INTENT(IN) :: N
        REAL(DP), INTENT(IN) :: X_MIN, X_MAX, Y_MIN, Y_MAX
        REAL(DP) :: AREA
        REAL(DP) :: CLIPPED(32,2)
        INTEGER :: N_CLIPPED
        INTEGER :: I

        CALL CLIP_POLYGON(POLY, N, X_MIN, X_MAX, Y_MIN, Y_MAX, CLIPPED, N_CLIPPED)
        IF (N_CLIPPED < 3) THEN
            AREA = 0.0_DP
        ELSE
            AREA = POLYGON_AREA(CLIPPED, N_CLIPPED)
        END IF
    END FUNCTION POLYRECT_INTERSECTION_AREA

    SUBROUTINE CLIP_POLYGON(POLY, N, X_MIN, X_MAX, Y_MIN, Y_MAX, OUT_POLY, OUT_N)
        IMPLICIT NONE
        REAL(DP), INTENT(IN) :: POLY(32,2)
        INTEGER, INTENT(IN) :: N
        REAL(DP), INTENT(IN) :: X_MIN, X_MAX, Y_MIN, Y_MAX
        REAL(DP), INTENT(OUT) :: OUT_POLY(32,2)
        INTEGER, INTENT(OUT) :: OUT_N
        REAL(DP) :: TMP1(32,2), TMP2(32,2)
        INTEGER :: N_TMP1, N_TMP2

        CALL CLIP_EDGE(POLY, N, 1, X_MIN, TMP1, N_TMP1)
        CALL CLIP_EDGE(TMP1, N_TMP1, -1, X_MAX, TMP2, N_TMP2)
        CALL CLIP_EDGE_Y(TMP2, N_TMP2, 1, Y_MIN, TMP1, N_TMP1)
        CALL CLIP_EDGE_Y(TMP1, N_TMP1, -1, Y_MAX, OUT_POLY, OUT_N)
    END SUBROUTINE CLIP_POLYGON

    SUBROUTINE CLIP_EDGE(POLY, N, SIGN, X_EDGE, OUT_POLY, OUT_N)
        IMPLICIT NONE
        REAL(DP), INTENT(IN) :: POLY(32,2)
        INTEGER, INTENT(IN) :: N, SIGN
        REAL(DP), INTENT(IN) :: X_EDGE
        REAL(DP), INTENT(OUT) :: OUT_POLY(32,2)
        INTEGER, INTENT(OUT) :: OUT_N
        INTEGER :: I, J
        REAL(DP) :: X0, Y0, X1, Y1, T, X_INT, Y_INT
        OUT_N = 0
        DO I = 1, N
            J = MOD(I, N) + 1
            X0 = POLY(I,1)
            Y0 = POLY(I,2)
            X1 = POLY(J,1)
            Y1 = POLY(J,2)
            IF (SIGN * (X1 - X_EDGE) >= 0) THEN
                IF (SIGN * (X0 - X_EDGE) < 0) THEN
                    T = (X_EDGE - X0) / (X1 - X0)
                    Y_INT = Y0 + T * (Y1 - Y0)
                    OUT_N = OUT_N + 1
                    OUT_POLY(OUT_N,1) = X_EDGE
                    OUT_POLY(OUT_N,2) = Y_INT
                END IF
                OUT_N = OUT_N + 1
                OUT_POLY(OUT_N,1) = X1
                OUT_POLY(OUT_N,2) = Y1
            ELSE IF (SIGN * (X0 - X_EDGE) >= 0) THEN
                T = (X_EDGE - X0) / (X1 - X0)
                Y_INT = Y0 + T * (Y1 - Y0)
                OUT_N = OUT_N + 1
                OUT_POLY(OUT_N,1) = X_EDGE
                OUT_POLY(OUT_N,2) = Y_INT
            END IF
        END DO
    END SUBROUTINE CLIP_EDGE

    SUBROUTINE CLIP_EDGE_Y(POLY, N, SIGN, Y_EDGE, OUT_POLY, OUT_N)
        IMPLICIT NONE
        REAL(DP), INTENT(IN) :: POLY(32,2)
        INTEGER, INTENT(IN) :: N, SIGN
        REAL(DP), INTENT(IN) :: Y_EDGE
        REAL(DP), INTENT(OUT) :: OUT_POLY(32,2)
        INTEGER, INTENT(OUT) :: OUT_N
        INTEGER :: I, J
        REAL(DP) :: X0, Y0, X1, Y1, T, X_INT, Y_INT
        OUT_N = 0
        DO I = 1, N
            J = MOD(I, N) + 1
            X0 = POLY(I,1)
            Y0 = POLY(I,2)
            X1 = POLY(J,1)
            Y1 = POLY(J,2)
            IF (SIGN * (Y1 - Y_EDGE) >= 0) THEN
                IF (SIGN * (Y0 - Y_EDGE) < 0) THEN
                    T = (Y_EDGE - Y0) / (Y1 - Y0)
                    X_INT = X0 + T * (X1 - X0)
                    OUT_N = OUT_N + 1
                    OUT_POLY(OUT_N,1) = X_INT
                    OUT_POLY(OUT_N,2) = Y_EDGE
                END IF
                OUT_N = OUT_N + 1
                OUT_POLY(OUT_N,1) = X1
                OUT_POLY(OUT_N,2) = Y1
            ELSE IF (SIGN * (Y0 - Y_EDGE) >= 0) THEN
                T = (Y_EDGE - Y0) / (Y1 - Y0)
                X_INT = X0 + T * (X1 - X0)
                OUT_N = OUT_N + 1
                OUT_POLY(OUT_N,1) = X_INT
                OUT_POLY(OUT_N,2) = Y_EDGE
            END IF
        END DO
    END SUBROUTINE CLIP_EDGE_Y

    FUNCTION POLYGON_AREA(POLY, N) RESULT(AREA)
        IMPLICIT NONE
        REAL(DP), INTENT(IN) :: POLY(32,2)
        INTEGER, INTENT(IN) :: N
        REAL(DP) :: AREA
        INTEGER :: I, J
        AREA = 0.0_DP
        DO I = 1, N
            J = MOD(I, N) + 1
            AREA = AREA + (POLY(I,1) * POLY(J,2) - POLY(J,1) * POLY(I,2))
        END DO
        AREA = 0.5_DP * ABS(AREA)
    END FUNCTION POLYGON_AREA

    PURE FUNCTION RECT_SLICE_RAW_OVERLAP_FACE(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, SLICE_POLY) RESULT(RAW)
        REAL(DP), INTENT(IN) :: X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX
        REAL(DP), INTENT(IN) :: SLICE_POLY(4, 3)
        REAL(DP) :: RAW
        REAL(DP) :: X0, Y0, WIDTH_H, Z_SLICE_MIN, Z_SLICE_MAX
        REAL(DP) :: NX, NY, TX, TY
        REAL(DP) :: ETA_1, ETA_2, ETA_3, ETA_4, ETA_MIN, ETA_MAX
        REAL(DP) :: ETA_S1, ETA_S2, ETA_S3, ETA_S4, ETA_SLICE_MIN, ETA_SLICE_MAX
        REAL(DP) :: XI_1, XI_2, XI_3, XI_4, XI_MIN, XI_MAX
        REAL(DP) :: XI_S1, XI_S2, XI_S3, XI_S4, XI_SLICE_MIN, XI_SLICE_MAX
        REAL(DP) :: RAW_H_ETA, RAW_H_XI, RAW_H
        REAL(DP) :: RAW_V
        REAL(DP), PARAMETER :: EPS = 1.0E-12_DP

        RAW = 0.0_DP

        X0 = 0.5_DP * (SLICE_POLY(1, 1) + SLICE_POLY(4, 1))
        Y0 = 0.5_DP * (SLICE_POLY(1, 2) + SLICE_POLY(4, 2))

        WIDTH_H = SQRT((SLICE_POLY(4, 1) - SLICE_POLY(1, 1))**2 + (SLICE_POLY(4, 2) - SLICE_POLY(1, 2))**2)
        IF (WIDTH_H <= EPS) RETURN

        NX = (SLICE_POLY(4, 1) - SLICE_POLY(1, 1)) / WIDTH_H
        NY = (SLICE_POLY(4, 2) - SLICE_POLY(1, 2)) / WIDTH_H
        TX = -NY
        TY = NX

        ETA_1 = NX * (X_MIN - X0) + NY * (Y_MIN - Y0)
        ETA_2 = NX * (X_MIN - X0) + NY * (Y_MAX - Y0)
        ETA_3 = NX * (X_MAX - X0) + NY * (Y_MIN - Y0)
        ETA_4 = NX * (X_MAX - X0) + NY * (Y_MAX - Y0)
        ETA_MIN = MIN(MIN(ETA_1, ETA_2), MIN(ETA_3, ETA_4))
        ETA_MAX = MAX(MAX(ETA_1, ETA_2), MAX(ETA_3, ETA_4))

        ETA_S1 = NX * (SLICE_POLY(1, 1) - X0) + NY * (SLICE_POLY(1, 2) - Y0)
        ETA_S2 = NX * (SLICE_POLY(2, 1) - X0) + NY * (SLICE_POLY(2, 2) - Y0)
        ETA_S3 = NX * (SLICE_POLY(3, 1) - X0) + NY * (SLICE_POLY(3, 2) - Y0)
        ETA_S4 = NX * (SLICE_POLY(4, 1) - X0) + NY * (SLICE_POLY(4, 2) - Y0)
        ETA_SLICE_MIN = MIN(MIN(ETA_S1, ETA_S2), MIN(ETA_S3, ETA_S4))
        ETA_SLICE_MAX = MAX(MAX(ETA_S1, ETA_S2), MAX(ETA_S3, ETA_S4))

        RAW_H_ETA = OVERLAP_1D(ETA_MIN, ETA_MAX, ETA_SLICE_MIN, ETA_SLICE_MAX)
        IF (RAW_H_ETA <= EPS) RETURN

        XI_1 = TX * (X_MIN - X0) + TY * (Y_MIN - Y0)
        XI_2 = TX * (X_MIN - X0) + TY * (Y_MAX - Y0)
        XI_3 = TX * (X_MAX - X0) + TY * (Y_MIN - Y0)
        XI_4 = TX * (X_MAX - X0) + TY * (Y_MAX - Y0)
        XI_MIN = MIN(MIN(XI_1, XI_2), MIN(XI_3, XI_4))
        XI_MAX = MAX(MAX(XI_1, XI_2), MAX(XI_3, XI_4))

        XI_S1 = TX * (SLICE_POLY(1, 1) - X0) + TY * (SLICE_POLY(1, 2) - Y0)
        XI_S2 = TX * (SLICE_POLY(2, 1) - X0) + TY * (SLICE_POLY(2, 2) - Y0)
        XI_S3 = TX * (SLICE_POLY(3, 1) - X0) + TY * (SLICE_POLY(3, 2) - Y0)
        XI_S4 = TX * (SLICE_POLY(4, 1) - X0) + TY * (SLICE_POLY(4, 2) - Y0)
        XI_SLICE_MIN = MIN(MIN(XI_S1, XI_S2), MIN(XI_S3, XI_S4))
        XI_SLICE_MAX = MAX(MAX(XI_S1, XI_S2), MAX(XI_S3, XI_S4))

        IF ((XI_SLICE_MAX - XI_SLICE_MIN) <= EPS) THEN
            IF ((XI_MIN > 0.0_DP) .OR. (XI_MAX < 0.0_DP)) RETURN
            RAW_H_XI = 1.0_DP
        ELSE
            RAW_H_XI = OVERLAP_1D(XI_MIN, XI_MAX, XI_SLICE_MIN, XI_SLICE_MAX)
            IF (RAW_H_XI <= EPS) RETURN
        END IF
        RAW_H = RAW_H_ETA * RAW_H_XI
        
        IF (RAW_H <= EPS) RETURN

        Z_SLICE_MIN = MIN(MIN(SLICE_POLY(1, 3), SLICE_POLY(2, 3)), MIN(SLICE_POLY(3, 3), SLICE_POLY(4, 3)))
        Z_SLICE_MAX = MAX(MAX(SLICE_POLY(1, 3), SLICE_POLY(2, 3)), MAX(SLICE_POLY(3, 3), SLICE_POLY(4, 3)))
        RAW_V = OVERLAP_1D(Z_MIN, Z_MAX, Z_SLICE_MIN, Z_SLICE_MAX)
        IF (RAW_V <= EPS) RETURN

        RAW = RAW_H * RAW_V
    END FUNCTION RECT_SLICE_RAW_OVERLAP_FACE

END MODULE HELPERS

MODULE RUN_CHEM_UTILS
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: CHEM_ALLOC, RUN_LEGACY_CHEM

    ! Primary chemistry workspace arrays
    DOUBLE PRECISION, ALLOCATABLE :: Y(:,:), YP(:,:)
    DOUBLE PRECISION, ALLOCATABLE :: RC(:,:), J(:,:), DJ(:,:), FL(:,:)
    DOUBLE PRECISION, ALLOCATABLE :: TEMP(:), H2O(:), O2(:), N2(:), SZA(:)
    DOUBLE PRECISION, ALLOCATABLE :: SOA(:), MOM(:), BR01(:), RO2(:)
    DOUBLE PRECISION, ALLOCATABLE :: P(:), L(:)

    ! CHEMCO: simple rate coefficient scalars
    DOUBLE PRECISION :: KRO2NO3, KDEC

    ! CHEMCO: complex rate coefficient scalars
    DOUBLE PRECISION :: FCC, FCD, FC1, K2I, FC2, FC7, FC8, K9I, FC9, FC10, KI, K13I
    DOUBLE PRECISION :: FC13, FC14, FC15, FC16, FCX

    ! CHEMCO: simple rate coefficient arrays (NCELL-sized, allocated in CHEM_ALLOC)
    DOUBLE PRECISION, ALLOCATABLE :: KRO2NO(:), KAPNO(:), KRO2HO2(:), KAPHO2(:)
    DOUBLE PRECISION, ALLOCATABLE :: KNO3AL(:), KALKOXY(:), KALKPXY(:), KIN(:)
    DOUBLE PRECISION, ALLOCATABLE :: KOUT2604(:), KOUT4608(:), KOUT2631(:), KOUT2635(:)
    DOUBLE PRECISION, ALLOCATABLE :: KOUT4610(:), KOUT2605(:), KOUT2630(:), KOUT2629(:)
    DOUBLE PRECISION, ALLOCATABLE :: KOUT2632(:), KOUT2637(:), KOUT3612(:), KOUT3613(:)
    DOUBLE PRECISION, ALLOCATABLE :: KOUT3442(:)

    ! CHEMCO: complex rate coefficient arrays
    DOUBLE PRECISION, ALLOCATABLE :: KC0(:), KCI(:), KRC(:), FC(:), KFPAN(:)
    DOUBLE PRECISION, ALLOCATABLE :: KD0(:), KDI(:), KRD(:), FD(:), KBPAN(:)
    DOUBLE PRECISION, ALLOCATABLE :: K10(:), K1I(:), KR1(:), F1(:), KMT01(:)
    DOUBLE PRECISION, ALLOCATABLE :: K20(:), KR2(:), Fa2(:), KMT02(:)
    DOUBLE PRECISION, ALLOCATABLE :: K30(:), K3I(:), KR3(:), FC3(:), F3(:), KMT03(:)
    DOUBLE PRECISION, ALLOCATABLE :: K40(:), K4I(:), KR4(:), FC4(:), Fa4(:), KMT04(:)
    DOUBLE PRECISION, ALLOCATABLE :: KMT05(:), KMT06(:)
    DOUBLE PRECISION, ALLOCATABLE :: K70(:), K7I(:), KR7(:), F7(:), KMT07(:)
    DOUBLE PRECISION, ALLOCATABLE :: K80(:), K8I(:), KR8(:), F8(:), KMT08(:)
    DOUBLE PRECISION, ALLOCATABLE :: K90(:), KR9(:), F9(:), KMT09(:)
    DOUBLE PRECISION, ALLOCATABLE :: K100(:), K10I(:), KR10(:), F10(:), KMT10(:)
    DOUBLE PRECISION, ALLOCATABLE :: K1(:), K3(:), K4(:), K2(:), KMT11(:)
    DOUBLE PRECISION, ALLOCATABLE :: K0(:), F(:), KMT12(:)
    DOUBLE PRECISION, ALLOCATABLE :: K130(:), KR13(:), F13(:), KMT13(:)
    DOUBLE PRECISION, ALLOCATABLE :: K140(:), K14I(:), KR14(:), F14(:), KMT14(:)
    DOUBLE PRECISION, ALLOCATABLE :: K150(:), K15I(:), KR15(:), F15(:), KMT15(:)
    DOUBLE PRECISION, ALLOCATABLE :: K160(:), K16I(:), KR16(:), F16(:), KMT16(:)
    DOUBLE PRECISION, ALLOCATABLE :: K170(:), K17I(:), KR17(:), FC17(:), F17(:), KMT17(:)

CONTAINS
    SUBROUTINE CALC_AEROSOL(Y, SOA, MOM)
        IMPLICIT NONE
        DOUBLE PRECISION, INTENT(IN) :: Y(:,:)
        DOUBLE PRECISION, INTENT(OUT) :: SOA(:)
        DOUBLE PRECISION, INTENT(OUT) :: MOM(:)
        DOUBLE PRECISION :: BGOAM

        BGOAM = 0.7

        SOA(:) = Y(:,204)*3.574E-10 + Y(:,205)*3.574E-10 + &
        Y(:,206)*3.059E-10 + Y(:,207)*3.126E-10 + Y(:,208)*3.093E-10 + &
        Y(:,209)*3.093E-10 + Y(:,210)*3.325E-10 + Y(:,211)*4.072E-10 + &
        Y(:,212)*2.860E-10 + Y(:,213)*3.391E-10 + Y(:,214)*2.310E-10 + &
        Y(:,215)*2.543E-10 + Y(:,216)*1.628E-10 + Y(:,219)*2.493E-10
        
        MOM(:) = Y(:,218) + BGOAM + SOA(:)

    END SUBROUTINE CALC_AEROSOL

    SUBROUTINE CHEM_ALLOC(NCELL, NSBOXM, NPP, NPC, NTC, NFL)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NCELL, NSBOXM, NPP, NPC, NTC, NFL

        ! Deallocate all workspace arrays (core and K arrays)
        IF (ALLOCATED(Y)) DEALLOCATE(Y)
        IF (ALLOCATED(YP)) DEALLOCATE(YP)
        IF (ALLOCATED(SOA)) DEALLOCATE(SOA)
        IF (ALLOCATED(MOM)) DEALLOCATE(MOM)
        IF (ALLOCATED(J)) DEALLOCATE(J)
        IF (ALLOCATED(DJ)) DEALLOCATE(DJ)
        IF (ALLOCATED(RC)) DEALLOCATE(RC)
        IF (ALLOCATED(FL)) DEALLOCATE(FL)

        IF (ALLOCATED(TEMP)) DEALLOCATE(TEMP)
        IF (ALLOCATED(H2O)) DEALLOCATE(H2O)
        IF (ALLOCATED(O2)) DEALLOCATE(O2)
        IF (ALLOCATED(N2)) DEALLOCATE(N2)
        IF (ALLOCATED(SZA)) DEALLOCATE(SZA)

        ! Deallocate all K workspace arrays
        IF (ALLOCATED(KRO2NO)) DEALLOCATE(KRO2NO)
        IF (ALLOCATED(KAPNO)) DEALLOCATE(KAPNO)
        IF (ALLOCATED(KRO2HO2)) DEALLOCATE(KRO2HO2)
        IF (ALLOCATED(KAPHO2)) DEALLOCATE(KAPHO2)
        IF (ALLOCATED(KNO3AL)) DEALLOCATE(KNO3AL)
        IF (ALLOCATED(KALKOXY)) DEALLOCATE(KALKOXY)
        IF (ALLOCATED(KALKPXY)) DEALLOCATE(KALKPXY)
        IF (ALLOCATED(KIN)) DEALLOCATE(KIN)
        IF (ALLOCATED(KOUT2604)) DEALLOCATE(KOUT2604)
        IF (ALLOCATED(KOUT4608)) DEALLOCATE(KOUT4608)
        IF (ALLOCATED(KOUT2631)) DEALLOCATE(KOUT2631)
        IF (ALLOCATED(KOUT2635)) DEALLOCATE(KOUT2635)
        IF (ALLOCATED(KOUT4610)) DEALLOCATE(KOUT4610)
        IF (ALLOCATED(KOUT2605)) DEALLOCATE(KOUT2605)
        IF (ALLOCATED(KOUT2630)) DEALLOCATE(KOUT2630)
        IF (ALLOCATED(KOUT2629)) DEALLOCATE(KOUT2629)
        IF (ALLOCATED(KOUT2632)) DEALLOCATE(KOUT2632)
        IF (ALLOCATED(KOUT2637)) DEALLOCATE(KOUT2637)
        IF (ALLOCATED(KOUT3612)) DEALLOCATE(KOUT3612)
        IF (ALLOCATED(KOUT3613)) DEALLOCATE(KOUT3613)
        IF (ALLOCATED(KOUT3442)) DEALLOCATE(KOUT3442)
        IF (ALLOCATED(KC0)) DEALLOCATE(KC0)
        IF (ALLOCATED(KCI)) DEALLOCATE(KCI)
        IF (ALLOCATED(KRC)) DEALLOCATE(KRC)
        IF (ALLOCATED(FC)) DEALLOCATE(FC)
        IF (ALLOCATED(KFPAN)) DEALLOCATE(KFPAN)
        IF (ALLOCATED(KD0)) DEALLOCATE(KD0)
        IF (ALLOCATED(KDI)) DEALLOCATE(KDI)
        IF (ALLOCATED(KRD)) DEALLOCATE(KRD)
        IF (ALLOCATED(FD)) DEALLOCATE(FD)
        IF (ALLOCATED(KBPAN)) DEALLOCATE(KBPAN)
        IF (ALLOCATED(K10)) DEALLOCATE(K10)
        IF (ALLOCATED(K1I)) DEALLOCATE(K1I)
        IF (ALLOCATED(KR1)) DEALLOCATE(KR1)
        IF (ALLOCATED(F1)) DEALLOCATE(F1)
        IF (ALLOCATED(KMT01)) DEALLOCATE(KMT01)
        IF (ALLOCATED(K20)) DEALLOCATE(K20)
        IF (ALLOCATED(KR2)) DEALLOCATE(KR2)
        IF (ALLOCATED(Fa2)) DEALLOCATE(Fa2)
        IF (ALLOCATED(KMT02)) DEALLOCATE(KMT02)
        IF (ALLOCATED(K30)) DEALLOCATE(K30)
        IF (ALLOCATED(K3I)) DEALLOCATE(K3I)
        IF (ALLOCATED(KR3)) DEALLOCATE(KR3)
        IF (ALLOCATED(FC3)) DEALLOCATE(FC3)
        IF (ALLOCATED(F3)) DEALLOCATE(F3)
        IF (ALLOCATED(KMT03)) DEALLOCATE(KMT03)
        IF (ALLOCATED(K40)) DEALLOCATE(K40)
        IF (ALLOCATED(K4I)) DEALLOCATE(K4I)
        IF (ALLOCATED(KR4)) DEALLOCATE(KR4)
        IF (ALLOCATED(FC4)) DEALLOCATE(FC4)
        IF (ALLOCATED(Fa4)) DEALLOCATE(Fa4)
        IF (ALLOCATED(KMT04)) DEALLOCATE(KMT04)
        IF (ALLOCATED(KMT05)) DEALLOCATE(KMT05)
        IF (ALLOCATED(KMT06)) DEALLOCATE(KMT06)
        IF (ALLOCATED(K70)) DEALLOCATE(K70)
        IF (ALLOCATED(K7I)) DEALLOCATE(K7I)
        IF (ALLOCATED(KR7)) DEALLOCATE(KR7)
        IF (ALLOCATED(F7)) DEALLOCATE(F7)
        IF (ALLOCATED(KMT07)) DEALLOCATE(KMT07)
        IF (ALLOCATED(K80)) DEALLOCATE(K80)
        IF (ALLOCATED(K8I)) DEALLOCATE(K8I)
        IF (ALLOCATED(KR8)) DEALLOCATE(KR8)
        IF (ALLOCATED(F8)) DEALLOCATE(F8)
        IF (ALLOCATED(KMT08)) DEALLOCATE(KMT08)
        IF (ALLOCATED(K90)) DEALLOCATE(K90)
        IF (ALLOCATED(KR9)) DEALLOCATE(KR9)
        IF (ALLOCATED(F9)) DEALLOCATE(F9)
        IF (ALLOCATED(KMT09)) DEALLOCATE(KMT09)
        IF (ALLOCATED(K100)) DEALLOCATE(K100)
        IF (ALLOCATED(K10I)) DEALLOCATE(K10I)
        IF (ALLOCATED(KR10)) DEALLOCATE(KR10)
        IF (ALLOCATED(F10)) DEALLOCATE(F10)
        IF (ALLOCATED(KMT10)) DEALLOCATE(KMT10)
        IF (ALLOCATED(K1)) DEALLOCATE(K1)
        IF (ALLOCATED(K3)) DEALLOCATE(K3)
        IF (ALLOCATED(K4)) DEALLOCATE(K4)
        IF (ALLOCATED(K2)) DEALLOCATE(K2)
        IF (ALLOCATED(KMT11)) DEALLOCATE(KMT11)
        IF (ALLOCATED(K0)) DEALLOCATE(K0)
        IF (ALLOCATED(F)) DEALLOCATE(F)
        IF (ALLOCATED(KMT12)) DEALLOCATE(KMT12)
        IF (ALLOCATED(K130)) DEALLOCATE(K130)
        IF (ALLOCATED(KR13)) DEALLOCATE(KR13)
        IF (ALLOCATED(F13)) DEALLOCATE(F13)
        IF (ALLOCATED(KMT13)) DEALLOCATE(KMT13)
        IF (ALLOCATED(K140)) DEALLOCATE(K140)
        IF (ALLOCATED(K14I)) DEALLOCATE(K14I)
        IF (ALLOCATED(KR14)) DEALLOCATE(KR14)
        IF (ALLOCATED(F14)) DEALLOCATE(F14)
        IF (ALLOCATED(KMT14)) DEALLOCATE(KMT14)
        IF (ALLOCATED(K150)) DEALLOCATE(K150)
        IF (ALLOCATED(K15I)) DEALLOCATE(K15I)
        IF (ALLOCATED(KR15)) DEALLOCATE(KR15)
        IF (ALLOCATED(F15)) DEALLOCATE(F15)
        IF (ALLOCATED(KMT15)) DEALLOCATE(KMT15)
        IF (ALLOCATED(K160)) DEALLOCATE(K160)
        IF (ALLOCATED(K16I)) DEALLOCATE(K16I)
        IF (ALLOCATED(KR16)) DEALLOCATE(KR16)
        IF (ALLOCATED(F16)) DEALLOCATE(F16)
        IF (ALLOCATED(KMT16)) DEALLOCATE(KMT16)
        IF (ALLOCATED(K170)) DEALLOCATE(K170)
        IF (ALLOCATED(K17I)) DEALLOCATE(K17I)
        IF (ALLOCATED(KR17)) DEALLOCATE(KR17)
        IF (ALLOCATED(FC17)) DEALLOCATE(FC17)
        IF (ALLOCATED(F17)) DEALLOCATE(F17)
        IF (ALLOCATED(KMT17)) DEALLOCATE(KMT17)
        IF (ALLOCATED(BR01)) DEALLOCATE(BR01)
        IF (ALLOCATED(RO2)) DEALLOCATE(RO2)
        IF (ALLOCATED(P)) DEALLOCATE(P)
        IF (ALLOCATED(L)) DEALLOCATE(L)
        
        ! Allocate all workspace arrays (core and K arrays) once per timestep, sized for NCELL
        ALLOCATE(Y(NCELL, NSBOXM), YP(NCELL, NSBOXM))
        ALLOCATE(SOA(NCELL), MOM(NCELL))
        ALLOCATE(J(NCELL, NPP), DJ(NCELL, NPC), RC(NCELL, NTC), FL(NCELL, NFL))
        ALLOCATE(BR01(NCELL), RO2(NCELL), P(NCELL), L(NCELL))
        ALLOCATE(TEMP(NCELL), H2O(NCELL), O2(NCELL), N2(NCELL), SZA(NCELL))
        ALLOCATE(KRO2NO(NCELL), KAPNO(NCELL), KRO2HO2(NCELL), KAPHO2(NCELL))
        ALLOCATE(KNO3AL(NCELL), KALKOXY(NCELL), KALKPXY(NCELL), KIN(NCELL))
        ALLOCATE(KOUT2604(NCELL), KOUT4608(NCELL), KOUT2631(NCELL), KOUT2635(NCELL))
        ALLOCATE(KOUT4610(NCELL), KOUT2605(NCELL), KOUT2630(NCELL), KOUT2629(NCELL))
        ALLOCATE(KOUT2632(NCELL), KOUT2637(NCELL), KOUT3612(NCELL), KOUT3613(NCELL))
        ALLOCATE(KOUT3442(NCELL))
        ALLOCATE(KC0(NCELL), KCI(NCELL), KRC(NCELL), FC(NCELL), KFPAN(NCELL))
        ALLOCATE(KD0(NCELL), KDI(NCELL), KRD(NCELL), FD(NCELL), KBPAN(NCELL))
        ALLOCATE(K10(NCELL), K1I(NCELL), KR1(NCELL), F1(NCELL), KMT01(NCELL))
        ALLOCATE(K20(NCELL), KR2(NCELL), Fa2(NCELL), KMT02(NCELL))
        ALLOCATE(K30(NCELL), K3I(NCELL), KR3(NCELL), FC3(NCELL), F3(NCELL), KMT03(NCELL))
        ALLOCATE(K40(NCELL), K4I(NCELL), KR4(NCELL), FC4(NCELL), Fa4(NCELL), KMT04(NCELL))
        ALLOCATE(KMT05(NCELL), KMT06(NCELL))
        ALLOCATE(K70(NCELL), K7I(NCELL), KR7(NCELL), F7(NCELL), KMT07(NCELL))
        ALLOCATE(K80(NCELL), K8I(NCELL), KR8(NCELL), F8(NCELL), KMT08(NCELL))
        ALLOCATE(K90(NCELL), KR9(NCELL), F9(NCELL), KMT09(NCELL))
        ALLOCATE(K100(NCELL), K10I(NCELL), KR10(NCELL), F10(NCELL), KMT10(NCELL))
        ALLOCATE(K1(NCELL), K3(NCELL), K4(NCELL), K2(NCELL), KMT11(NCELL))
        ALLOCATE(K0(NCELL), F(NCELL), KMT12(NCELL))
        ALLOCATE(K130(NCELL), KR13(NCELL), F13(NCELL), KMT13(NCELL))
        ALLOCATE(K140(NCELL), K14I(NCELL), KR14(NCELL), F14(NCELL), KMT14(NCELL))
        ALLOCATE(K150(NCELL), K15I(NCELL), KR15(NCELL), F15(NCELL), KMT15(NCELL))
        ALLOCATE(K160(NCELL), K16I(NCELL), KR16(NCELL), F16(NCELL), KMT16(NCELL))
        ALLOCATE(K170(NCELL), K17I(NCELL), KR17(NCELL), FC17(NCELL), F17(NCELL), KMT17(NCELL))

        ! Initialize all workspace arrays to zero
        Y = 0.0D0
        YP = 0.0D0
        SOA = 0.0D0
        MOM = 0.0D0
        J = 0.0D0
        DJ = 0.0D0
        RC = 0.0D0
        FL = 0.0D0
        P = 0.0D0
        L = 0.0D0
        BR01 = 0.0D0
        RO2 = 0.0D0
        TEMP = 0.0D0
        H2O = 0.0D0
        O2 = 0.0D0
        N2 = 0.0D0
        SZA = 0.0D0

        KRO2NO = 0.0D0
        KAPNO = 0.0D0
        KRO2HO2 = 0.0D0
        KAPHO2 = 0.0D0
        KNO3AL = 0.0D0
        KALKOXY = 0.0D0
        KALKPXY = 0.0D0
        KIN = 0.0D0
        KOUT2604 = 0.0D0
        KOUT4608 = 0.0D0
        KOUT2631 = 0.0D0
        KOUT2635 = 0.0D0
        KOUT4610 = 0.0D0
        KOUT2605 = 0.0D0
        KOUT2630 = 0.0D0
        KOUT2629 = 0.0D0
        KOUT2632 = 0.0D0
        KOUT2637 = 0.0D0
        KOUT3612 = 0.0D0
        KOUT3613 = 0.0D0
        KOUT3442 = 0.0D0
        KC0 = 0.0D0
        KCI = 0.0D0
        KRC = 0.0D0
        FC = 0.0D0
        KFPAN = 0.0D0
        KD0 = 0.0D0
        KDI = 0.0D0
        KRD = 0.0D0
        FD = 0.0D0
        KBPAN = 0.0D0
        K10 = 0.0D0
        K1I = 0.0D0
        KR1 = 0.0D0
        F1 = 0.0D0
        KMT01 = 0.0D0
        K20 = 0.0D0
        KR2 = 0.0D0
        Fa2 = 0.0D0
        KMT02 = 0.0D0
        K30 = 0.0D0
        K3I = 0.0D0
        KR3 = 0.0D0
        FC3 = 0.0D0
        F3 = 0.0D0
        KMT03 = 0.0D0
        K40 = 0.0D0
        K4I = 0.0D0
        KR4 = 0.0D0
        FC4 = 0.0D0
        Fa4 = 0.0D0
        KMT04 = 0.0D0
        KMT05 = 0.0D0
        KMT06 = 0.0D0
        K70 = 0.0D0
        K7I = 0.0D0
        KR7 = 0.0D0
        F7 = 0.0D0
        KMT07 = 0.0D0
        K80 = 0.0D0
        K8I = 0.0D0
        KR8 = 0.0D0
        F8 = 0.0D0
        KMT08 = 0.0D0
        K90 = 0.0D0
        KR9 = 0.0D0
        F9 = 0.0D0
        KMT09 = 0.0D0
        K100 = 0.0D0
        K10I = 0.0D0
        KR10 = 0.0D0
        F10 = 0.0D0
        KMT10 = 0.0D0
        K1 = 0.0D0
        K3 = 0.0D0
        K4 = 0.0D0
        K2 = 0.0D0
        KMT11 = 0.0D0
        K0 = 0.0D0
        F = 0.0D0
        KMT12 = 0.0D0
        K130 = 0.0D0
        KR13 = 0.0D0
        F13 = 0.0D0
        KMT13 = 0.0D0
        K140 = 0.0D0
        K14I = 0.0D0
        KR14 = 0.0D0
        F14 = 0.0D0
        KMT14 = 0.0D0
        K150 = 0.0D0
        K15I = 0.0D0
        KR15 = 0.0D0
        F15 = 0.0D0
        KMT15 = 0.0D0
        K160 = 0.0D0
        K16I = 0.0D0
        KR16 = 0.0D0
        F16 = 0.0D0
        KMT16 = 0.0D0
        K170 = 0.0D0
        K17I = 0.0D0
        KR17 = 0.0D0
        FC17 = 0.0D0
        F17 = 0.0D0
        KMT17 = 0.0D0
    END SUBROUTINE CHEM_ALLOC

    SUBROUTINE CHEMCO(TEMP, N2, O2, M, RC)
        
        IMPLICIT NONE
        DOUBLE PRECISION, INTENT(IN)  :: TEMP(:), N2(:), O2(:), M(:)
        DOUBLE PRECISION, INTENT(OUT) :: RC(:,:)
        DOUBLE PRECISION :: R
        INTEGER :: I

        !         ! K-array workspace accessed from module-level (allocated in CHEM_ALLOC)
        !         ! All rate coefficients available via host association

                R = 8.314
            
                !     SIMPLE RATE COEFFICIENTS                     
                                                                        
                KRO2NO(:)  = 2.70D-12*EXP(360/TEMP(:)) 
                KAPNO(:)   = 7.50D-12*EXP(290/TEMP(:)) 
                KRO2NO3    = 2.30D-12 
                KRO2HO2(:) = 2.91D-13*EXP(1300/TEMP(:)) 
                KAPHO2(:)  = 5.20D-13*EXP(980/TEMP(:)) 
                KNO3AL(:)  = 1.44D-12*EXP(-1862/TEMP(:)) 
                KDEC       = 1.0D+06
                KALKOXY(:) = 3.70D-14*EXP(-460/TEMP(:))*O2(:) 
                KALKPXY(:) = 1.80D-14*EXP(-260/TEMP(:))*O2(:) 
                BR01(:)    = (0.156 + 9.77D+08*EXP(-6415/TEMP(:))) 

                KIN(:)      = 6.2E-03*MOM(:)
                KOUT2604(:) = 4.34*EXP(-7776/(R*TEMP(:)))
                KOUT4608(:) = 4.34*EXP(-9765/(R*TEMP(:)))
                KOUT2631(:) = 4.34*EXP(-14500/(R*TEMP(:)))
                KOUT2635(:) = 4.34*EXP(-12541/(R*TEMP(:)))
                KOUT4610(:) = 4.34*EXP(-10513/(R*TEMP(:)))
                KOUT2605(:) = 4.34*EXP(-8879/(R*TEMP(:)))
                KOUT2630(:) = 4.34*EXP(-12639/(R*TEMP(:)))
                KOUT2629(:) = 4.34*EXP(-4954/(R*TEMP(:)))
                KOUT2632(:) = 4.34*EXP(-3801/(R*TEMP(:)))
                KOUT2637(:) = 4.34*EXP(-16752/(R*TEMP(:)))
                KOUT3612(:) = 4.34*EXP(-8362/(R*TEMP(:)))
                KOUT3613(:) = 4.34*EXP(-11003/(R*TEMP(:)))
                KOUT3442(:) = 4.34*EXP(-12763/(R*TEMP(:)))

                    
                !    COMPLEX RATE COEFFICIENTS                    
                                                                            
                !    KFPAN                                                          
                KC0(:)     = 3.28D-28*M(:)*(TEMP(:)/300)**(-7.1)
                KCI(:)     = 1.125D-11*(TEMP(:)/300)**(-1.105)
                KRC(:)     = KC0(:)/KCI(:)
                FCC        = 0.30
                FC(:)      = 10**(LOG10(FCC)/(1+(LOG10(KRC(:)))**2)) 
                KFPAN(:)   = (KC0(:)*KCI(:))*FC(:)/(KC0(:)+KCI(:)) 

                !    KBPAN                                                   
                KD0(:)     = 1.1D-05*M(:)*EXP(-10100/TEMP(:)) 
                KDI(:)     = 1.9D+17*EXP(-14100/TEMP(:))  
                KRD(:)     = KD0(:)/KDI(:)    
                FCD        = 0.30       
                FD(:)      = 10**(LOG10(FCD)/(1+(LOG10(KRD(:)))**2)) 
                KBPAN(:)   = (KD0(:)*KDI(:))*FD(:)/(KD0(:)+KDI(:)) 
                                                                    
                !     KMT01                                                   
                K10(:)     = 9.00D-32*M(:)*(TEMP(:)/300)**(-1.5)
                K1I(:)     = 3.00D-11*(TEMP(:)/300)**0.3    
                KR1(:)     = K10(:)/K1I(:)    
                FC1        = 0.6 
                F1(:)      = 10**(LOG10(FC1)/(1+(LOG10(KR1(:)))**2)) 
                KMT01(:)   = (K10(:)*K1I(:))*F1(:)/(K10+K1I(:)) 
                                                                            
                !     KMT02                                                   
                K20(:)     = 9.00D-32*((TEMP(:)/300)**(-2.0))*M(:) 
                K2I        = 2.20D-11
                KR2(:)     = K20(:)/K2I    
                FC2        = 0.6 
                Fa2(:)     = 10**(LOG10(FC2)/(1+(LOG10(KR2(:)))**2)) 
                KMT02(:)   = (K20(:)*K2I)*Fa2(:)/(K20(:)+K2I) 
                                                                            
                !      KMT03  : NO2      + NO3     = N2O5                               
                !    IUPAC 2001                                                       
                K30(:)     = 2.70D-30*M(:)*(TEMP(:)/300)**(-3.4)
                K3I(:)     = 2.00D-12*(TEMP(:)/300)**0.2    
                KR3(:)     = K30(:)/K3I(:)   
                FC3(:)     = (EXP(-TEMP(:)/250) + EXP(-1050/TEMP(:))) 
                F3(:)      = 10**(LOG10(FC3(:))/(1+(LOG10(KR3(:)))**2)) 
                KMT03(:)   = (K30(:)*K3I(:))*F3(:)/(K30(:)+K3I(:)) 
                                                                    
                !     KMT04  : N2O5               = NO2     + NO3                     
                ! IUPAC 1997/2001                                                 
                K40(:)     = (2.20D-03*M(:)*(TEMP(:)/300)**(-4.34))*(EXP(-11080/TEMP(:)))
                K4I(:)     = (9.70D+14*(TEMP(:)/300)**0.1)*EXP(-11080/TEMP(:))    
                KR4(:)     = K40(:)/K4I(:)    
                FC4(:)     = (EXP(-TEMP(:)/250) + EXP(-1050/TEMP(:)))
                Fa4(:)     = 10**(LOG10(FC4(:))/(1+(LOG10(KR4(:)))**2)) 
                KMT04(:)   = (K40(:)*K4I(:))*Fa4(:)/(K40(:)+K4I(:))       
                                                                
                !       KMT05                                                   
                KMT05(:)   = 1 + ((0.6*M(:))/(2.687D+19*(273/TEMP(:)))) 
                                                                            
                !    KMT06                                                   
                KMT06(:)   = 1 + (1.40D-21*EXP(2200/TEMP(:))*H2O(:)) 
                                                                            
                !    KMT07  : OH       + NO      = HONO                              
                !    IUPAC 2001                                                      
                K70(:)     = 7.00D-31*M(:)*(TEMP(:)/300)**(-2.6)
                K7I(:)     = 3.60D-11*(TEMP(:)/300)**0.1    
                KR7(:)     = K70(:)/K7I(:)  
                FC7        = 0.6  
                F7(:)      = 10**(LOG10(FC7)/(1+(LOG10(KR7(:)))**2)) 
                KMT07(:)   = (K70(:)*K7I(:))*F7(:)/(K70(:)+K7I(:)) 
                                                                            
                ! NASA 2000                                                           
                !    KMT08                                                    
                K80(:)     = 2.50D-30*((TEMP(:)/300)**(-4.4))*M(:) 
                K8I(:)     = 1.60D-11 
                KR8(:)     = K80(:)/K8I(:)
                FC8        = 0.6 
                F8(:)      = 10**(LOG10(FC8)/(1+(LOG10(KR8(:)))**2)) 
                KMT08(:)   = (K80(:)*K8I(:))*F8(:)/(K80(:)+K8I(:)) 
                                                                            
                !    KMT09  : HO2      + NO2     = HO2NO2                            
                !    IUPAC 1997/2001                                                 
                K90(:)     = 1.80D-31*M(:)*(TEMP(:)/300)**(-3.2)
                K9I        = 4.70D-12    
                KR9(:)     = K90(:)/K9I    
                FC9        = 0.6 
                F9(:)      = 10**(LOG10(FC9)/(1+(LOG10(KR9(:)))**2)) 
                KMT09(:)   = (K90(:)*K9I)*F9(:)/(K90(:)+K9I) 
                                                                            
                ! KMT10  : HO2NO2             = HO2     + NO2                     
                ! IUPAC 2001                                                      

                K100(:)    = 4.10D-05*M(:)*EXP(-10650/TEMP(:)) 
                K10I(:)    = 5.70D+15*EXP(-11170/TEMP(:))   
                KR10(:)    = K100(:)/K10I(:)    
                FC10       = 0.5 
                F10(:)     = 10**(LOG10(FC10)/(1+(LOG10(KR10(:)))**2)) 
                KMT10(:)   = (K100(:)*K10I(:))*F10(:)/(K100(:)+K10I(:)) 
                                                                            
                !   KMT11  : OH       + HNO3    = H2O + NO3                     
                !   IUPAC 2001                                                      
                K1(:)      = 7.20D-15*EXP(785/TEMP(:)) 
                K3(:)      = 1.90D-33*EXP(725/TEMP(:)) 
                K4(:)      = 4.10D-16*EXP(1440/TEMP(:)) 
                K2(:)      = (K3(:)*M(:))/(1+(K3(:)*M(:)/K4(:))) 
                KMT11(:)   = K1(:) + K2(:) 
                                                                    
                ! KMT12 : OH    +   SO2  =  HSO3                                  
                ! IUPAC 2003                                                      
                K0(:)      = 3.0D-31*((TEMP(:)/300)**(-3.3))*M(:) 
                KI         = 1.5D-12 
                KR1(:)     = K0(:)/KI 
                FCX        = 0.6 
                F(:)       = 10**(LOG10(FCX)/(1+(LOG10(KR1(:)))**2)) 
                KMT12(:)   = (K0(:)*KI*F(:))/(K0(:)+KI) 
                                                                    
                ! KMT13  : CH3O2    + NO2     = CH3O2NO2                           
                ! IUPAC 2003                                                       
                K130(:)     = 1.20D-30*((TEMP(:)/300)**(-6.9))*M(:) 
                K13I        = 1.80D-11 
                KR13(:)     = K130(:)/K13I 
                FC13        = 0.36 
                F13(:)      = 10**(LOG10(FC13)/(1+(LOG10(KR13(:)))**2)) 
                KMT13(:)    = (K130(:)*K13I)*F13(:)/(K130(:)+K13I) 
                                                                            
                !  KMT14  : CH3O2NO2           = CH3O2   + NO2                      
                !  IUPAC 2001                                                       
                K140(:)     = 9.00D-05*EXP(-9690/TEMP(:))*M(:) 
                K14I(:)     = 1.10D+16*EXP(-10560/TEMP(:)) 
                KR14(:)     = K140(:)/K14I(:)
                FC14        = 0.36 
                F14(:)      = 10**(LOG10(FC14)/(1+(LOG10(KR14(:)))**2)) 
                KMT14(:)    = (K140(:)*K14I(:))*F14(:)/(K140(:)+K14I(:)) 
                                                                
                ! KMT15  :    OH  +  C2H4  =                                       
                ! IUPAC 2001                                                      
                K150(:)     = 6.00D-29*((TEMP(:)/298)**(-4.0))*M(:) 
                K15I(:)     = 9.00D-12*((TEMP(:)/298)**(-1.1)) 
                KR15(:)     = K150(:)/K15I(:)
                FC15        = 0.7
                F15(:)      = 10**(LOG10(FC15)/(1+(LOG10(KR15(:)))**2)) 
                KMT15(:)    = (K150(:)*K15I(:))*F15(:)/(K150(:)+K15I(:)) 
                                                                    
                ! KMT16  :  OH  +  C3H6         
                ! IUPAC 2003                                                     
                K160(:)     = 3.00D-27*((TEMP(:)/298)**(-3.0))*M(:) 
                K16I(:)     = 2.80D-11*((TEMP(:)/298)**(-1.3)) 
                KR16(:)     = K160(:)/K16I(:) 
                FC16        = 0.5 
                F16(:)      = 10**(LOG10(FC16)/(1+(LOG10(KR16(:)))**2)) 
                KMT16(:)    = (K160(:)*K16I(:))*F16(:)/(K160(:)+K16I(:)) 

                ! KMT17                                                   
                K170(:)     = 5.00D-30*((TEMP(:)/298)**(-1.5))*M(:) 
                K17I(:)     = 9.40D-12*EXP(-700/TEMP(:)) 
                KR17(:)     = K170(:)/K17I(:)
                FC17(:)     = (EXP(-TEMP(:)/580) + EXP(-2320/TEMP(:))) 
                F17(:)      = 10**(LOG10(FC17(:))/(1+(LOG10(KR17(:)))**2)) 
                KMT17(:)    = (K170(:) * K17I(:)) * F17(:)/(K170(:) + K17I(:)) 

                ! LIST OF ALL REACTIONS
                ! Reaction (1) O = O3                                                             
                RC(:,1) = 5.60D-34*O2(:)*N2(:)*((TEMP(:)/300)**(-2.6))

                ! Reaction (2) O = O3                                                             
                RC(:,2) = 6.00D-34*O2(:)*O2(:)*((TEMP(:)/300)**(-2.6))

                ! Reaction (3) O + O3 =                                                           
                RC(:,3) = 8.00D-12*EXP(-2060/TEMP(:))         

                ! Reaction (4) O + NO = NO2                                                       
                RC(:,4) = KMT01(:)                          

                ! Reaction (5) O + NO2 = NO                                                       
                RC(:,5) = 5.50D-12*EXP(188/TEMP(:))           

                ! Reaction (6) O + NO2 = NO3                                                      
                RC(:,6) = KMT02(:)                            

                ! Reaction (7) O1D = O                                                            
                RC(:,7) = 3.20D-11*O2(:)*EXP(67/TEMP(:))         

                ! Reaction (8) O1D = O                                                            
                RC(:,8) = 1.80D-11*N2(:)*EXP(107/TEMP(:))        

                ! Reaction (9) NO + O3 = NO2                                                      
                RC(:,9) = 1.40D-12*EXP(-1310/TEMP(:))         

                ! Reaction (10) NO2 + O3 = NO3                                                     
                RC(:,10) = 1.40D-13*EXP(-2470/TEMP(:))         

                ! Reaction (11) NO + NO = NO2 + NO2                                                
                RC(:,11) = 3.30D-39*EXP(530/TEMP(:))*O2(:)        

                ! Reaction (12) NO + NO3 = NO2 + NO2                                               
                RC(:,12) = 1.80D-11*EXP(110/TEMP(:))           

                ! Reaction (13) NO2 + NO3 = NO + NO2                                               
                RC(:,13) = 4.50D-14*EXP(-1260/TEMP(:))         

                ! Reaction (14) NO2 + NO3 = N2O5                                                   
                RC(:,14) = KMT03(:)                            

                ! Reaction (15) N2O5 = NO2 + NO3                                                   
                RC(:,15) = KMT04(:)                            

                ! Reaction (16) O1D = OH + OH                                                      
                RC(:,16) = 2.20D-10                     

                ! Reaction (17) OH + O3 = HO2                                                      
                RC(:,17) = 1.70D-12*EXP(-940/TEMP(:))          

                ! Reaction (18) OH + H2 = HO2                                                      
                RC(:,18) = 7.70D-12*EXP(-2100/TEMP(:))         

                ! Reaction (19) OH + CO = HO2                                                      
                RC(:,19) = 1.30D-13*KMT05(:)                   

                ! Reaction (20) OH + H2O2 = HO2                                                    
                RC(:,20) = 2.90D-12*EXP(-160/TEMP(:))          

                ! Reaction (21) HO2 + O3 = OH                                                      
                RC(:,21) = 2.03D-16*((TEMP(:)/300)**4.57)*EXP(693/TEMP(:))  

                ! Reaction (22) OH + HO2 =                                                         
                RC(:,22) = 4.80D-11*EXP(250/TEMP(:))           

                ! Reaction (23) HO2 + HO2 = H2O2                                                   
                RC(:,23) = 2.20D-13*KMT06(:)*EXP(600/TEMP(:))     

                ! Reaction (24) HO2 + HO2 = H2O2                                                   
                RC(:,24) = 1.90D-33*M(:)*KMT06(:)*EXP(980/TEMP(:))   

                ! Reaction (25) OH + NO = HONO                                                     
                RC(:,25) = KMT07(:)                            

                ! Reaction (26) NO2 = HONO                                                         
                RC(:,26) = 5.0D-07                          

                ! Reaction (27) OH + NO2 = HNO3                                                    
                RC(:,27) = KMT08(:)                            

                ! Reaction (28) OH + NO3 = HO2 + NO2                                               
                RC(:,28) = 2.00D-11                         

                ! Reaction (29) HO2 + NO = OH + NO2                                                
                RC(:,29) = 3.60D-12*EXP(270/TEMP(:))           

                ! Reaction (30) HO2 + NO2 = HO2NO2                                                 
                RC(:,30) = KMT09(:)                            

                ! Reaction (31) HO2NO2 = HO2 + NO2                                                 
                RC(:,31) = KMT10(:)                            

                ! Reaction (32) OH + HO2NO2 = NO2                                                  
                RC(:,32) = 1.90D-12*EXP(270/TEMP(:))           

                ! Reaction (33) HO2 + NO3 = OH + NO2                                               
                RC(:,33) = 4.00D-12                         

                ! Reaction (34) OH + HONO = NO2                                                    
                RC(:,34) = 2.50D-12*EXP(-260/TEMP(:))          

                ! Reaction (35) OH + HNO3 = NO3                                                    
                RC(:,35) = KMT11(:)                            

                ! Reaction (36) O + SO2 = SO3                                                      
                RC(:,36) = 4.00D-32*EXP(-1000/TEMP(:))*M(:)       

                ! Reaction (37) OH + SO2 = HSO3                                                    
                RC(:,37) = KMT12(:)                            

                ! Reaction (38) HSO3 = HO2 + SO3                                                   
                RC(:,38) = 1.30D-12*EXP(-330/TEMP(:))*O2(:)       

                ! Reaction (39) HNO3 = NA                                                          
                RC(:,39) = 6.00D-06                         

                ! Reaction (40) N2O5 = NA + NA                                                     
                RC(:,40) = 4.00D-05                       

                ! Reaction (41) SO3 = SA                                                           
                RC(:,41) = 1.20D-15*H2O(:)                     

                ! Reaction (42) OH + CH4 = CH3O2                                                   
                RC(:,42) = 9.65D-20*TEMP(:)**2.58*EXP(-1082/TEMP(:)) 

                ! Reaction (43) OH + C2H6 = C2H5O2                                                 
                RC(:,43) = 1.52D-17*TEMP(:)**2*EXP(-498/TEMP(:)) 

                ! Reaction (44) OH + C3H8 = IC3H7O2                                                
                RC(:,44) = 1.55D-17*TEMP(:)**2*EXP(-61/TEMP(:))*0.736  

                ! Reaction (45) OH + C3H8 = RN10O2                                                 
                RC(:,45) = 1.55D-17*TEMP(:)**2*EXP(-61/TEMP(:))*0.264  

                ! Reaction (46) OH + NC4H10 = RN13O2                                               
                RC(:,46) = 1.69D-17*TEMP(:)**2*EXP(145/TEMP(:))  

                ! Reaction (47) OH + C2H4 = HOCH2CH2O2                                             
                RC(:,47) = KMT15(:)                        

                ! Reaction (48) OH + C3H6 = RN9O2                                                  
                RC(:,48) = KMT16(:)                        

                ! Reaction (49) OH + TBUT2ENE = RN12O2                                             
                RC(:,49) = 1.01D-11*EXP(550/TEMP(:))       

                ! Reaction (50) NO3 + C2H4 = NRN6O2                                                
                RC(:,50) = 2.10D-16                     

                ! Reaction (51) NO3 + C3H6 = NRN9O2                                                
                RC(:,51) = 9.40D-15                     

                ! Reaction (52) NO3 + TBUT2ENE = NRN12O2                                           
                RC(:,52) = 3.90D-13                     

                ! Reaction (53) O3 + C2H4 = HCHO + CO + HO2 + OH                                   
                RC(:,53) = 9.14D-15*EXP(-2580/TEMP(:))*0.13  

                ! Reaction (54) O3 + C2H4 = HCHO + HCOOH                                           
                RC(:,54) = 9.14D-15*EXP(-2580/TEMP(:))*0.87  

                ! Reaction (55) O3 + C3H6 = HCHO + CO + CH3O2 + OH                                 
                RC(:,55) = 5.51D-15*EXP(-1878/TEMP(:))*0.36  

                ! Reaction (56) O3 + C3H6 = HCHO + CH3CO2H                                         
                RC(:,56) = 5.51D-15*EXP(-1878/TEMP(:))*0.64  

                ! Reaction (57) O3 + TBUT2ENE = CH3CHO + CO + CH3O2 + OH                           
                RC(:,57) = 6.64D-15*EXP(-1059/TEMP(:))*0.69 

                ! Reaction (58) O3 + TBUT2ENE = CH3CHO + CH3CO2H                                   
                RC(:,58) = 6.64D-15*EXP(-1059/TEMP(:))*0.31 

                ! Reaction (59) OH + C5H8 = RU14O2                                                 
                RC(:,59) = 2.70D-11*EXP(390/TEMP(:))       

                ! Reaction (60) NO3 + C5H8 = NRU14O2                                               
                RC(:,60) = 3.15D-12*EXP(-450/TEMP(:))      

                ! Reaction (61) O3 + C5H8 = UCARB10 + CO + HO2 + OH                                
                RC(:,61) = 1.03D-14*EXP(-1995/TEMP(:))*0.27 

                ! Reaction (62) O3 + C5H8 = UCARB10 + HCOOH                                        
                RC(:,62) = 1.03D-14*EXP(-1995/TEMP(:))*0.73 

                ! Reaction (63) APINENE + OH = RTN28O2                                             
                RC(:,63) = 1.20D-11*EXP(444/TEMP(:))           

                ! Reaction (64) APINENE + NO3 = NRTN28O2                                           
                RC(:,64) = 1.19D-12*EXP(490/TEMP(:))           

                ! Reaction (65) APINENE + O3 = OH + RTN26O2                                        
                RC(:,65) = 1.01D-15*EXP(-732/TEMP(:))*0.80  

                ! Reaction (66) APINENE + O3 = TNCARB26 + H2O2                                     
                RC(:,66) = 1.01D-15*EXP(-732/TEMP(:))*0.075  

                ! Reaction (67) APINENE + O3 = RCOOH25                                             
                RC(:,67) = 1.01D-15*EXP(-732/TEMP(:))*0.125  

                ! Reaction (68) BPINENE + OH = RTX28O2                                             
                RC(:,68) = 2.38D-11*EXP(357/TEMP(:)) 

                ! Reaction (69) BPINENE + NO3 = NRTX28O2                                           
                RC(:,69) = 2.51D-12 

                ! Reaction (70) BPINENE + O3 =  RTX24O2 + OH                                       
                RC(:,70) = 1.50D-17*0.35 

                ! Reaction (71) BPINENE + O3 =  HCHO + TXCARB24 + H2O2                             
                RC(:,71) = 1.50D-17*0.20 

                ! Reaction (72) BPINENE + O3 =  HCHO + TXCARB22                                    
                RC(:,72) = 1.50D-17*0.25 

                ! Reaction (73) BPINENE + O3 =  TXCARB24 + CO                                      
                RC(:,73) = 1.50D-17*0.20 

                ! Reaction (74) C2H2 + OH = HCOOH + CO + HO2                                       
                RC(:,74) = KMT17(:)*0.364 

                ! Reaction (75) C2H2 + OH = CARB3 + OH                                             
                RC(:,75) = KMT17(:)*0.636 

                ! Reaction (76) BENZENE + OH = RA13O2                                              
                RC(:,76) = 2.33D-12*EXP(-193/TEMP(:))*0.47 

                ! Reaction (77) BENZENE + OH = AROH14 + HO2                                        
                RC(:,77) = 2.33D-12*EXP(-193/TEMP(:))*0.53 

                ! Reaction (78) TOLUENE + OH = RA16O2                                              
                RC(:,78) = 1.81D-12*EXP(338/TEMP(:))*0.82 

                ! Reaction (79) TOLUENE + OH = AROH17 + HO2                                        
                RC(:,79) = 1.81D-12*EXP(338/TEMP(:))*0.18 

                ! Reaction (80) OXYL + OH = RA19AO2                                                
                RC(:,80) = 1.36D-11*0.70 

                ! Reaction (81) OXYL + OH = RA19CO2                                                
                RC(:,81) = 1.36D-11*0.30 

                ! Reaction (82) OH + HCHO = HO2 + CO                                               
                RC(:,82) = 1.20D-14*TEMP(:)*EXP(287/TEMP(:))  

                ! Reaction (83) OH + CH3CHO = CH3CO3                                               
                RC(:,83) = 5.55D-12*EXP(311/TEMP(:))             

                ! Reaction (84) OH + C2H5CHO = C2H5CO3                                             
                RC(:,84) = 1.96D-11                                

                ! Reaction (85) NO3 + HCHO = HO2 + CO + HNO3                                       
                RC(:,85) = 5.80D-16                  

                ! Reaction (86) NO3 + CH3CHO = CH3CO3 + HNO3                                       
                RC(:,86) = KNO3AL(:)                   

                ! Reaction (87) NO3 + C2H5CHO = C2H5CO3 + HNO3                                     
                RC(:,87) = KNO3AL(:)*2.4             

                ! Reaction (88) OH + CH3COCH3 = RN8O2                                              
                RC(:,88) = 5.34D-18*TEMP(:)**2*EXP(-230/TEMP(:)) 

                ! Reaction (89) MEK + OH = RN11O2                                                  
                RC(:,89) = 3.24D-18*TEMP(:)**2*EXP(414/TEMP(:))

                ! Reaction (90) OH + CH3OH = HO2 + HCHO                                            
                RC(:,90) = 6.01D-18*TEMP(:)**2*EXP(170/TEMP(:))  

                ! Reaction (91) OH + C2H5OH = CH3CHO + HO2                                         
                RC(:,91) = 6.18D-18*TEMP(:)**2*EXP(532/TEMP(:))*0.887 

                ! Reaction (92) OH + C2H5OH = HOCH2CH2O2                                           
                RC(:,92) = 6.18D-18*TEMP(:)**2*EXP(532/TEMP(:))*0.113 

                ! Reaction (93) NPROPOL + OH = C2H5CHO + HO2                                       
                RC(:,93) = 5.53D-12*0.49 

                ! Reaction (94) NPROPOL + OH = RN9O2                                               
                RC(:,94) = 5.53D-12*0.51 

                ! Reaction (95) OH + IPROPOL = CH3COCH3 + HO2                                      
                RC(:,95) = 4.06D-18*TEMP(:)**2*EXP(788/TEMP(:))*0.86 

                ! Reaction (96) OH + IPROPOL = RN9O2                                               
                RC(:,96) = 4.06D-18*TEMP(:)**2*EXP(788/TEMP(:))*0.14 

                ! Reaction (97) HCOOH + OH = HO2                                                   
                RC(:,97) = 4.50D-13 

                ! Reaction (98) CH3CO2H + OH = CH3O2                                               
                RC(:,98) = 8.00D-13 

                ! Reaction (99) OH + CH3CL = CH3O2                                                 
                RC(:,99) = 7.33D-18*TEMP(:)**2*EXP(-809/TEMP(:))   

                ! Reaction (100) OH + CH2CL2 = CH3O2                                                
                RC(:,100) = 6.14D-18*TEMP(:)**2*EXP(-389/TEMP(:))   

                ! Reaction (101) OH + CHCL3 = CH3O2                                                 
                RC(:,101) = 1.80D-18*TEMP(:)**2*EXP(-129/TEMP(:))   

                ! Reaction (102) OH + CH3CCL3 = C2H5O2                                              
                RC(:,102) = 2.25D-18*TEMP(:)**2*EXP(-910/TEMP(:))   

                ! Reaction (103) OH + TCE = HOCH2CH2O2                                              
                RC(:,103) = 9.64D-12*EXP(-1209/TEMP(:))         

                ! Reaction (104) OH + TRICLETH = HOCH2CH2O2                                         
                RC(:,104) = 5.63D-13*EXP(427/TEMP(:))            

                ! Reaction (105) OH + CDICLETH = HOCH2CH2O2                                         
                RC(:,105) = 1.94D-12*EXP(90/TEMP(:))            

                ! Reaction (106) OH + TDICLETH = HOCH2CH2O2                                         
                RC(:,106) = 1.01D-12*EXP(250/TEMP(:))           

                ! Reaction (107) CH3O2 + NO = HCHO + HO2 + NO2                                      
                RC(:,107) = 3.00D-12*EXP(280/TEMP(:))*0.999 

                ! Reaction (108) C2H5O2 + NO = CH3CHO + HO2 + NO2                                   
                RC(:,108) = 2.60D-12*EXP(365/TEMP(:))*0.991 

                ! Reaction (109) RN10O2 + NO = C2H5CHO + HO2 + NO2                                  
                RC(:,109) = 2.80D-12*EXP(360/TEMP(:))*0.980 

                ! Reaction (110) IC3H7O2 + NO = CH3COCH3 + HO2 + NO2                                
                RC(:,110) = 2.70D-12*EXP(360/TEMP(:))*0.958 

                ! Reaction (111) RN13O2 + NO = CH3CHO + C2H5O2 + NO2                                
                RC(:,111) = KRO2NO(:)*0.917*BR01(:)       

                ! Reaction (112) RN13O2 + NO = CARB11A + HO2 + NO2                                  
                RC(:,112) = KRO2NO(:)*0.917*(1-BR01(:))   

                ! Reaction (113) RN16O2 + NO = RN15AO2 + NO2                                        
                RC(:,113) = KRO2NO(:)*0.877                 

                ! Reaction (114) RN19O2 + NO = RN18AO2 + NO2                                        
                RC(:,114) = KRO2NO(:)*0.788                 

                ! Reaction (115) RN13AO2 + NO = RN12O2 + NO2                                        
                RC(:,115) = KRO2NO(:)                       

                ! Reaction (116) RN16AO2 + NO = RN15O2 + NO2                                        
                RC(:,116) = KRO2NO(:)                       

                ! Reaction (117) RA13O2 + NO = CARB3 + UDCARB8 + HO2 + NO2                          
                RC(:,117) = KRO2NO(:)*0.918       

                ! Reaction (118) RA16O2 + NO = CARB3 + UDCARB11 + HO2 + NO2                         
                RC(:,118) = KRO2NO(:)*0.889*0.7 

                ! Reaction (119) RA16O2 + NO = CARB6 + UDCARB8 + HO2 + NO2                          
                RC(:,119) = KRO2NO(:)*0.889*0.3 

                ! Reaction (120) RA19AO2 + NO = CARB3 + UDCARB14 + HO2 + NO2                        
                RC(:,120) = KRO2NO(:)*0.862       

                ! Reaction (121) RA19CO2 + NO = CARB9 + UDCARB8 + HO2 + NO2                         
                RC(:,121) = KRO2NO(:)*0.862       

                ! Reaction (122) HOCH2CH2O2 + NO = HCHO + HCHO + HO2 + NO2                          
                RC(:,122) = KRO2NO(:)*0.995*0.776  

                ! Reaction (123) HOCH2CH2O2 + NO = HOCH2CHO + HO2 + NO2                             
                RC(:,123) = KRO2NO(:)*0.995*0.224  

                ! Reaction (124) RN9O2 + NO = CH3CHO + HCHO + HO2 + NO2                             
                RC(:,124) = KRO2NO(:)*0.979     

                ! Reaction (125) RN12O2 + NO = CH3CHO + CH3CHO + HO2 + NO2                          
                RC(:,125) = KRO2NO(:)*0.959     

                ! Reaction (126) RN15O2 + NO = C2H5CHO + CH3CHO + HO2 + NO2                         
                RC(:,126) = KRO2NO(:)*0.936     

                ! Reaction (127) RN18O2 + NO = C2H5CHO + C2H5CHO + HO2 + NO2                        
                RC(:,127) = KRO2NO(:)*0.903     

                ! Reaction (128) RN15AO2 + NO = CARB13 + HO2 + NO2                                  
                RC(:,128) = KRO2NO(:)*0.975     

                ! Reaction (129) RN18AO2 + NO = CARB16 + HO2 + NO2                                  
                RC(:,129) = KRO2NO(:)*0.946     

                ! Reaction (130) CH3CO3 + NO = CH3O2 + NO2                                          
                RC(:,130) = KAPNO(:)                      

                ! Reaction (131) C2H5CO3 + NO = C2H5O2 + NO2                                        
                RC(:,131) = KAPNO(:)                      

                ! Reaction (132) HOCH2CO3 + NO = HO2 + HCHO + NO2                                   
                RC(:,132) = KAPNO(:)                      

                ! Reaction (133) RN8O2 + NO = CH3CO3 + HCHO + NO2                                   
                RC(:,133) = KRO2NO(:)                     

                ! Reaction (134) RN11O2 + NO = CH3CO3 + CH3CHO + NO2                                
                RC(:,134) = KRO2NO(:)                     

                ! Reaction (135) RN14O2 + NO = C2H5CO3 + CH3CHO + NO2                               
                RC(:,135) = KRO2NO(:)                     

                ! Reaction (136) RN17O2 + NO = RN16AO2 + NO2                                        
                RC(:,136) = KRO2NO(:)                     

                ! Reaction (137) RU14O2 + NO = UCARB12 + HO2 +  NO2                                 
                RC(:,137) = KRO2NO(:)*0.900*0.252  

                ! Reaction (138) RU14O2 + NO = UCARB10 + HCHO + HO2 + NO2                           
                RC(:,138) = KRO2NO(:)*0.900*0.748 

                ! Reaction (139) RU12O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
                RC(:,139) = KRO2NO(:)*0.7         

                ! Reaction (140) RU12O2 + NO = CARB7 + CO + HO2 + NO2                               
                RC(:,140) = KRO2NO(:)*0.3         

                ! Reaction (141) RU10O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
                RC(:,141) = KRO2NO(:)*0.670      

                ! Reaction (142) RU10O2 + NO = CARB6 + HCHO + HO2 + NO2                             
                RC(:,142) = KRO2NO(:)*0.295         

                ! Reaction (143) RU10O2 + NO = CARB7 + HCHO + HO2 + NO2                             
                RC(:,143) = KRO2NO(:)*0.035          

                ! Reaction (144) NRN6O2 + NO = HCHO + HCHO + NO2 + NO2                              
                RC(:,144) = KRO2NO(:)                 

                ! Reaction (145) NRN9O2 + NO = CH3CHO + HCHO + NO2 + NO2                            
                RC(:,145) = KRO2NO(:)                 

                ! Reaction (146) NRN12O2 + NO = CH3CHO + CH3CHO + NO2 + NO2                         
                RC(:,146) = KRO2NO(:)                 

                ! Reaction (147) NRU14O2 + NO = NUCARB12 + HO2 + NO2                                
                RC(:,147) = KRO2NO(:)                 

                ! Reaction (148) NRU12O2 + NO = NOA + CO + HO2 + NO2                                
                RC(:,148) = KRO2NO(:)                 

                ! Reaction (149) RTN28O2 + NO = TNCARB26 + HO2 + NO2                                
                RC(:,149) = KRO2NO(:)*0.767*0.915  

                ! Reaction (150) RTN28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
                RC(:,150) = KRO2NO(:)*0.767*0.085  

                ! Reaction (151) NRTN28O2 + NO = TNCARB26 + NO2 + NO2                               
                RC(:,151) = KRO2NO(:)                  

                ! Reaction (152) RTN26O2 + NO = RTN25O2 + NO2                                       
                RC(:,152) = KAPNO(:)                   

                ! Reaction (153) RTN25O2 + NO = RTN24O2 + NO2                                       
                RC(:,153) = KRO2NO(:)*0.840        

                ! Reaction (154) RTN24O2 + NO = RTN23O2 + NO2                                       
                RC(:,154) = KRO2NO(:)                   

                ! Reaction (155) RTN23O2 + NO = CH3COCH3 + RTN14O2 + NO2                            
                RC(:,155) = KRO2NO(:)                  

                ! Reaction (156) RTN14O2 + NO = HCHO + TNCARB10 + HO2 + NO2                         
                RC(:,156) = KRO2NO(:)               

                ! Reaction (157) RTN10O2 + NO = RN8O2 + CO + NO2                                    
                RC(:,157) = KRO2NO(:)               

                ! Reaction (158) RTX28O2 + NO = TXCARB24 + HCHO + HO2 + NO2                         
                RC(:,158) = KRO2NO(:)*0.767*0.915  

                ! Reaction (159) RTX28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
                RC(:,159) = KRO2NO(:)*0.767*0.085  

                ! Reaction (160) NRTX28O2 + NO = TXCARB24 + HCHO + NO2 + NO2                        
                RC(:,160) = KRO2NO(:)            

                ! Reaction (161) RTX24O2 + NO = TXCARB22 + HO2 + NO2                                
                RC(:,161) = KRO2NO(:)*0.843*0.6  

                ! Reaction (162) RTX24O2 + NO = CH3COCH3 + RN13AO2 + HCHO + NO2                     
                RC(:,162) = KRO2NO(:)*0.843*0.4  

                ! Reaction (163) RTX22O2 + NO = CH3COCH3 + RN13O2 + NO2                             
                RC(:,163) = KRO2NO(:)*0.700         

                ! Reaction (164) CH3O2    + NO2     = CH3O2NO2                                      
                RC(:,164) = KMT13(:)         

                ! Reaction (165) CH3O2NO2           = CH3O2   + NO2                                 
                RC(:,165) = KMT14(:)         

                ! Reaction (166) CH3O2 + NO = CH3NO3                                                
                RC(:,166) = 3.00D-12*EXP(280/TEMP(:))*0.001 

                ! Reaction (167) C2H5O2 + NO = C2H5NO3                                              
                RC(:,167) = 2.60D-12*EXP(365/TEMP(:))*0.009 

                ! Reaction (168) RN10O2 + NO = RN10NO3                                              
                RC(:,168) = 2.80D-12*EXP(360/TEMP(:))*0.020 

                ! Reaction (169) IC3H7O2 + NO = IC3H7NO3                                            
                RC(:,169) = 2.70D-12*EXP(360/TEMP(:))*0.042 

                ! Reaction (170) RN13O2 + NO = RN13NO3                                              
                RC(:,170) = KRO2NO(:)*0.083                 

                ! Reaction (171) RN16O2 + NO = RN16NO3                                              
                RC(:,171) = KRO2NO(:)*0.123                 

                ! Reaction (172) RN19O2 + NO = RN19NO3                                              
                RC(:,172) = KRO2NO(:)*0.212                 

                ! Reaction (173) HOCH2CH2O2 + NO = HOC2H4NO3                                        
                RC(:,173) = KRO2NO(:)*0.005                 

                ! Reaction (174) RN9O2 + NO = RN9NO3                                                
                RC(:,174) = KRO2NO(:)*0.021                 

                ! Reaction (175) RN12O2 + NO = RN12NO3                                              
                RC(:,175) = KRO2NO(:)*0.041                 

                ! Reaction (176) RN15O2 + NO = RN15NO3                                              
                RC(:,176) = KRO2NO(:)*0.064                 

                ! Reaction (177) RN18O2 + NO = RN18NO3                                              
                RC(:,177) = KRO2NO(:)*0.097                 

                ! Reaction (178) RN15AO2 + NO = RN15NO3                                             
                RC(:,178) = KRO2NO(:)*0.025                 

                ! Reaction (179) RN18AO2 + NO = RN18NO3                                             
                RC(:,179) = KRO2NO(:)*0.054                 

                ! Reaction (180) RU14O2 + NO = RU14NO3                                              
                RC(:,180) = KRO2NO(:)*0.100                 

                ! Reaction (181) RA13O2 + NO = RA13NO3                                              
                RC(:,181) = KRO2NO(:)*0.082                 

                ! Reaction (182) RA16O2 + NO = RA16NO3                                              
                RC(:,182) = KRO2NO(:)*0.111                 

                ! Reaction (183) RA19AO2 + NO = RA19NO3                                             
                RC(:,183) = KRO2NO(:)*0.138                 

                ! Reaction (184) RA19CO2 + NO = RA19NO3                                             
                RC(:,184) = KRO2NO(:)*0.138                 

                ! Reaction (185) RTN28O2 + NO = RTN28NO3                                            
                RC(:,185) = KRO2NO(:)*0.233        

                ! Reaction (186) RTN25O2 + NO = RTN25NO3                                            
                RC(:,186) = KRO2NO(:)*0.160        

                ! Reaction (187) RTX28O2 + NO = RTX28NO3                                            
                RC(:,187) = KRO2NO(:)*0.233        

                ! Reaction (188) RTX24O2 + NO = RTX24NO3                                            
                RC(:,188) = KRO2NO(:)*0.157        

                ! Reaction (189) RTX22O2 + NO = RTX22NO3                                            
                RC(:,189) = KRO2NO(:)*0.300        

                ! Reaction (190) CH3O2 + NO3 = HCHO + HO2 + NO2                                     
                RC(:,190) = KRO2NO3*0.40          

                ! Reaction (191) C2H5O2 + NO3 = CH3CHO + HO2 + NO2                                  
                RC(:,191) = KRO2NO3               

                ! Reaction (192) RN10O2 + NO3 = C2H5CHO + HO2 + NO2                                 
                RC(:,192) = KRO2NO3               

                ! Reaction (193) IC3H7O2 + NO3 = CH3COCH3 + HO2 + NO2                               
                RC(:,193) = KRO2NO3               

                ! Reaction (194) RN13O2 + NO3 = CH3CHO + C2H5O2 + NO2                               
                RC(:,194) = KRO2NO3*BR01(:)     

                ! Reaction (195) RN13O2 + NO3 = CARB11A + HO2 + NO2                                 
                RC(:,195) = KRO2NO3*(1-BR01(:)) 

                ! Reaction (196) RN16O2 + NO3 = RN15AO2 + NO2                                       
                RC(:,196) = KRO2NO3               

                ! Reaction (197) RN19O2 + NO3 = RN18AO2 + NO2                                       
                RC(:,197) = KRO2NO3               

                ! Reaction (198) RN13AO2 + NO3 = RN12O2 + NO2                                       
                RC(:,198) = KRO2NO3                      

                ! Reaction (199) RN16AO2 + NO3 = RN15O2 + NO2                                       
                RC(:,199) = KRO2NO3                      

                ! Reaction (200) RA13O2 + NO3 = CARB3 + UDCARB8 + HO2 + NO2                         
                RC(:,200) = KRO2NO3            

                ! Reaction (201) RA16O2 + NO3 = CARB3 + UDCARB11 + HO2 + NO2                        
                RC(:,201) = KRO2NO3*0.7     

                ! Reaction (202) RA16O2 + NO3 = CARB6 + UDCARB8 + HO2 + NO2                         
                RC(:,202) = KRO2NO3*0.3     

                ! Reaction (203) RA19AO2 + NO3 = CARB3 + UDCARB14 + HO2 + NO2                       
                RC(:,203) = KRO2NO3           

                ! Reaction (204) RA19CO2 + NO3 = CARB9 + UDCARB8 + HO2 + NO2                        
                RC(:,204) = KRO2NO3           

                ! Reaction (205) HOCH2CH2O2 + NO3 = HCHO + HCHO + HO2 + NO2                         
                RC(:,205) = KRO2NO3*0.776  

                ! Reaction (206) HOCH2CH2O2 + NO3 = HOCH2CHO + HO2 + NO2                            
                RC(:,206) = KRO2NO3*0.224  

                ! Reaction (207) RN9O2 + NO3 = CH3CHO + HCHO + HO2 + NO2                            
                RC(:,207) = KRO2NO3         

                ! Reaction (208) RN12O2 + NO3 = CH3CHO + CH3CHO + HO2 + NO2                         
                RC(:,208) = KRO2NO3         

                ! Reaction (209) RN15O2 + NO3 = C2H5CHO + CH3CHO + HO2 + NO2                        
                RC(:,209) = KRO2NO3         

                ! Reaction (210) RN18O2 + NO3 = C2H5CHO + C2H5CHO + HO2 + NO2                       
                RC(:,210) = KRO2NO3         

                ! Reaction (211) RN15AO2 + NO3 = CARB13 + HO2 + NO2                                 
                RC(:,211) = KRO2NO3         

                ! Reaction (212) RN18AO2 + NO3 = CARB16 + HO2 + NO2                                 
                RC(:,212) = KRO2NO3         

                ! Reaction (213) CH3CO3 + NO3 = CH3O2 + NO2                                         
                RC(:,213) = KRO2NO3*1.60          

                ! Reaction (214) C2H5CO3 + NO3 = C2H5O2 + NO2                                       
                RC(:,214) = KRO2NO3*1.60          

                ! Reaction (215) HOCH2CO3 + NO3 = HO2 + HCHO + NO2                                  
                RC(:,215) = KRO2NO3*1.60         

                ! Reaction (216) RN8O2 + NO3 = CH3CO3 + HCHO + NO2                                  
                RC(:,216) = KRO2NO3               

                ! Reaction (217) RN11O2 + NO3 = CH3CO3 + CH3CHO + NO2                               
                RC(:,217) = KRO2NO3               

                ! Reaction (218) RN14O2 + NO3 = C2H5CO3 + CH3CHO + NO2                              
                RC(:,218) = KRO2NO3               

                ! Reaction (219) RN17O2 + NO3 = RN16AO2 + NO2                                       
                RC(:,219) = KRO2NO3               

                ! Reaction (220) RU14O2 + NO3 = UCARB12 + HO2 + NO2                                 
                RC(:,220) = KRO2NO3*0.032     

                ! Reaction (221) RU14O2 + NO3 = UCARB10 + HCHO + HO2 + NO2                          
                RC(:,221) = KRO2NO3*0.968     

                ! Reaction (222) RU12O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
                RC(:,222) = KRO2NO3*0.7         

                ! Reaction (223) RU12O2 + NO3 = CARB7 + CO + HO2 + NO2                              
                RC(:,223) = KRO2NO3*0.3         

                ! Reaction (224) RU10O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
                RC(:,224) = KRO2NO3*0.7         

                ! Reaction (225) RU10O2 + NO3 = CARB6 + HCHO + HO2 + NO2                            
                RC(:,225) = KRO2NO3*0.3         

                ! Reaction (226) RU10O2 + NO3 = CARB7 + HCHO + HO2 + NO2                            
                RC(:,226) = KRO2NO3*0.0        

                ! Reaction (227) NRN6O2 + NO3 = HCHO + HCHO + NO2 + NO2                             
                RC(:,227) = KRO2NO3               

                ! Reaction (228) NRN9O2 + NO3 = CH3CHO + HCHO + NO2 + NO2                           
                RC(:,228) = KRO2NO3               

                ! Reaction (229) NRN12O2 + NO3 = CH3CHO + CH3CHO + NO2 + NO2                        
                RC(:,229) = KRO2NO3               

                ! Reaction (230) NRU14O2 + NO3 = NUCARB12 + HO2 + NO2                               
                RC(:,230) = KRO2NO3               

                ! Reaction (231) NRU12O2 + NO3 = NOA + CO + HO2 + NO2                               
                RC(:,231) = KRO2NO3               

                ! Reaction (232) RTN28O2 + NO3 = TNCARB26 + HO2 + NO2                               
                RC(:,232) = KRO2NO3                

                ! Reaction (233) NRTN28O2 + NO3 = TNCARB26 + NO2 + NO2                              
                RC(:,233) = KRO2NO3                

                ! Reaction (234) RTN26O2 + NO3 = RTN25O2 + NO2                                      
                RC(:,234) = KRO2NO3*1.60                   

                ! Reaction (235) RTN25O2 + NO3 = RTN24O2 + NO2                                      
                RC(:,235) = KRO2NO3                 

                ! Reaction (236) RTN24O2 + NO3 = RTN23O2 + NO2                                      
                RC(:,236) = KRO2NO3                   

                ! Reaction (237) RTN23O2 + NO3 = CH3COCH3 + RTN14O2 + NO2                           
                RC(:,237) = KRO2NO3                 

                ! Reaction (238) RTN14O2 + NO3 = HCHO + TNCARB10 + HO2 + NO2                        
                RC(:,238) = KRO2NO3             

                ! Reaction (239) RTN10O2 + NO3 = RN8O2 + CO + NO2                                   
                RC(:,239) = KRO2NO3               

                ! Reaction (240) RTX28O2 + NO3 = TXCARB24 + HCHO + HO2 + NO2                        
                RC(:,240) = KRO2NO3             

                ! Reaction (241) RTX24O2 + NO3 = TXCARB22 + HO2 + NO2                               
                RC(:,241) = KRO2NO3             

                ! Reaction (242) RTX22O2 + NO3 = CH3COCH3 + RN13O2 + NO2                            
                RC(:,242) = KRO2NO3             

                ! Reaction (243) NRTX28O2 + NO3 = TXCARB24 + HCHO + NO2 + NO2                       
                RC(:,243) = KRO2NO3            

                ! Reaction (244) CH3O2 + HO2 = CH3OOH                                               
                RC(:,244) = 4.10D-13*EXP(790/TEMP(:))  

                ! Reaction (245) C2H5O2 + HO2 = C2H5OOH                                             
                RC(:,245) = 7.50D-13*EXP(700/TEMP(:))  

                ! Reaction (246) RN10O2 + HO2 = RN10OOH                                             
                RC(:,246) = KRO2HO2(:)*0.520           

                ! Reaction (247) IC3H7O2 + HO2 = IC3H7OOH                                           
                RC(:,247) = KRO2HO2(:)*0.520           

                ! Reaction (248) RN13O2 + HO2 = RN13OOH                                             
                RC(:,248) = KRO2HO2(:)*0.625           

                ! Reaction (249) RN16O2 + HO2 = RN16OOH                                             
                RC(:,249) = KRO2HO2(:)*0.706           

                ! Reaction (250) RN19O2 + HO2 = RN19OOH                                             
                RC(:,250) = KRO2HO2(:)*0.770           

                ! Reaction (251) RN13AO2 + HO2 = RN13OOH                                            
                RC(:,251) = KRO2HO2(:)*0.625           

                ! Reaction (252) RN16AO2 + HO2 = RN16OOH                                            
                RC(:,252) = KRO2HO2(:)*0.706           

                ! Reaction (253) RA13O2 + HO2 = RA13OOH                                             
                RC(:,253) = KRO2HO2(:)*0.770           

                ! Reaction (254) RA16O2 + HO2 = RA16OOH                                             
                RC(:,254) = KRO2HO2(:)*0.820           

                ! Reaction (255) RA19AO2 + HO2 = RA19OOH                                            
                RC(:,255) = KRO2HO2(:)*0.859           

                ! Reaction (256) RA19CO2 + HO2 = RA19OOH                                            
                RC(:,256) = KRO2HO2(:)*0.859           

                ! Reaction (257) HOCH2CH2O2 + HO2 = HOC2H4OOH                                       
                RC(:,257) = 2.03D-13*EXP(1250/TEMP(:)) 

                ! Reaction (258) RN9O2 + HO2 = RN9OOH                                               
                RC(:,258) = KRO2HO2(:)*0.520           

                ! Reaction (259) RN12O2 + HO2 = RN12OOH                                             
                RC(:,259) = KRO2HO2(:)*0.625           

                ! Reaction (260) RN15O2 + HO2 = RN15OOH                                             
                RC(:,260) = KRO2HO2(:)*0.706           

                ! Reaction (261) RN18O2 + HO2 = RN18OOH                                             
                RC(:,261) = KRO2HO2(:)*0.770           

                ! Reaction (262) RN15AO2 + HO2 = RN15OOH                                            
                RC(:,262) = KRO2HO2(:)*0.706           

                ! Reaction (263) RN18AO2 + HO2 = RN18OOH                                            
                RC(:,263) = KRO2HO2(:)*0.770           

                ! Reaction (264) CH3CO3 + HO2 = CH3CO3H                                             
                RC(:,264) = KAPHO2(:)*0.560                  

                ! Reaction (265) C2H5CO3 + HO2 = C2H5CO3H                                           
                RC(:,265) = KAPHO2(:)*0.560                  

                ! Reaction (266) HOCH2CO3 + HO2 = HOCH2CO3H                                         
                RC(:,266) = KAPHO2(:)*0.560                 

                ! Reaction (267) RN8O2 + HO2 = RN8OOH                                               
                RC(:,267) = KRO2HO2(:)*0.520           

                ! Reaction (268) RN11O2 + HO2 = RN11OOH                                             
                RC(:,268) = KRO2HO2(:)*0.625           

                ! Reaction (269) RN14O2 + HO2 = RN14OOH                                             
                RC(:,269) = KRO2HO2(:)*0.706           

                ! Reaction (270) RN17O2 + HO2 = RN17OOH                                             
                RC(:,270) = KRO2HO2(:)*0.770           

                ! Reaction (271) RU14O2 + HO2 = RU14OOH                                             
                RC(:,271) = KRO2HO2(:)*0.770           

                ! Reaction (272) RU12O2 + HO2 = RU12OOH                                             
                RC(:,272) = KRO2HO2(:)*0.706           

                ! Reaction (273) RU10O2 + HO2 = RU10OOH                                             
                RC(:,273) = KRO2HO2(:)*0.625           

                ! Reaction (274) NRN6O2 + HO2 = NRN6OOH                                             
                RC(:,274) = KRO2HO2(:)*0.387         

                ! Reaction (275) NRN9O2 + HO2 = NRN9OOH                                             
                RC(:,275) = KRO2HO2(:)*0.520         

                ! Reaction (276) NRN12O2 + HO2 = NRN12OOH                                           
                RC(:,276) = KRO2HO2(:)*0.625         

                ! Reaction (277) NRU14O2 + HO2 = NRU14OOH                                           
                RC(:,277) = KRO2HO2(:)*0.706         

                ! Reaction (278) NRU12O2 + HO2 = NRU12OOH                                           
                RC(:,278) = KRO2HO2(:)*0.706        

                ! Reaction (279) RTN28O2 + HO2 = RTN28OOH                                           
                RC(:,279) = KRO2HO2(:)*0.914         

                ! Reaction (280) NRTN28O2 + HO2 = NRTN28OOH                                         
                RC(:,280) = KRO2HO2(:)*0.914         

                ! Reaction (281) RTN26O2 + HO2 = RTN26OOH                                           
                RC(:,281) = KAPHO2(:)*0.56                     

                ! Reaction (282) RTN25O2 + HO2 = RTN25OOH                                           
                RC(:,282) = KRO2HO2(:)*0.890       

                ! Reaction (283) RTN24O2 + HO2 = RTN24OOH                                           
                RC(:,283) = KRO2HO2(:)*0.890       

                ! Reaction (284) RTN23O2 + HO2 = RTN23OOH                                           
                RC(:,284) = KRO2HO2(:)*0.890       

                ! Reaction (285) RTN14O2 + HO2 = RTN14OOH                                           
                RC(:,285) = KRO2HO2(:)*0.770       

                ! Reaction (286) RTN10O2 + HO2 = RTN10OOH                                           
                RC(:,286) = KRO2HO2(:)*0.706       

                ! Reaction (287) RTX28O2 + HO2 = RTX28OOH                                           
                RC(:,287) = KRO2HO2(:)*0.914       

                ! Reaction (288) RTX24O2 + HO2 = RTX24OOH                                           
                RC(:,288) = KRO2HO2(:)*0.890       

                ! Reaction (289) RTX22O2 + HO2 = RTX22OOH                                           
                RC(:,289) = KRO2HO2(:)*0.890       

                ! Reaction (290) NRTX28O2 + HO2 = NRTX28OOH                                         
                RC(:,290) = KRO2HO2(:)*0.914       

                ! Reaction (291) CH3O2 = HCHO + HO2                                                 
                RC(:,291) = 1.82D-13*EXP(416/TEMP(:))*0.33*RO2(:)  

                ! Reaction (292) CH3O2 = HCHO                                                       
                RC(:,292) = 1.82D-13*EXP(416/TEMP(:))*0.335*RO2(:) 

                ! Reaction (293) CH3O2 = CH3OH                                                      
                RC(:,293) = 1.82D-13*EXP(416/TEMP(:))*0.335*RO2(:) 

                ! Reaction (294) C2H5O2 = CH3CHO + HO2                                              
                RC(:,294) = 3.10D-13*0.6*RO2(:)             

                ! Reaction (295) C2H5O2 = CH3CHO                                                    
                RC(:,295) = 3.10D-13*0.2*RO2(:)             

                ! Reaction (296) C2H5O2 = C2H5OH                                                    
                RC(:,296) = 3.10D-13*0.2*RO2(:)             

                ! Reaction (297) RN10O2 = C2H5CHO + HO2                                             
                RC(:,297) = 6.00D-13*0.6*RO2(:)             

                ! Reaction (298) RN10O2 = C2H5CHO                                                   
                RC(:,298) = 6.00D-13*0.2*RO2(:)             

                ! Reaction (299) RN10O2 = NPROPOL                                                   
                RC(:,299) = 6.00D-13*0.2*RO2(:)             

                ! Reaction (300) IC3H7O2 = CH3COCH3 + HO2                                           
                RC(:,300) = 4.00D-14*0.6*RO2(:)             

                ! Reaction (301) IC3H7O2 = CH3COCH3                                                 
                RC(:,301) = 4.00D-14*0.2*RO2(:)             

                ! Reaction (302) IC3H7O2 = IPROPOL                                                  
                RC(:,302) = 4.00D-14*0.2*RO2(:)             

                ! Reaction (303) RN13O2 = CH3CHO + C2H5O2                                           
                RC(:,303) = 2.50D-13*RO2(:)*BR01(:)       

                ! Reaction (304) RN13O2 = CARB11A + HO2                                             
                RC(:,304) = 2.50D-13*RO2(:)*(1-BR01(:))   

                ! Reaction (305) RN13AO2 = RN12O2                                                   
                RC(:,305) = 8.80D-13*RO2(:)                 

                ! Reaction (306) RN16AO2 = RN15O2                                                   
                RC(:,306) = 8.80D-13*RO2(:)                 

                ! Reaction (307) RA13O2 = CARB3 + UDCARB8 + HO2                                     
                RC(:,307) = 8.80D-13*RO2(:)                 

                ! Reaction (308) RA16O2 = CARB3 + UDCARB11 + HO2                                    
                RC(:,308) = 8.80D-13*RO2(:)*0.7          

                ! Reaction (309) RA16O2 = CARB6 + UDCARB8 + HO2                                     
                RC(:,309) = 8.80D-13*RO2(:)*0.3          

                ! Reaction (310) RA19AO2 = CARB3 + UDCARB14 + HO2                                   
                RC(:,310) = 8.80D-13*RO2(:)                 

                ! Reaction (311) RA19CO2 = CARB3 + UDCARB14 + HO2                                   
                RC(:,311) = 8.80D-13*RO2(:)                 

                ! Reaction (312) RN16O2 = RN15AO2                                                   
                RC(:,312) = 2.50D-13*RO2(:)                 

                ! Reaction (313) RN19O2 = RN18AO2                                                   
                RC(:,313) = 2.50D-13*RO2(:)                 

                ! Reaction (314) HOCH2CH2O2 = HCHO + HCHO + HO2                                     
                RC(:,314) = 2.00D-12*RO2(:)*0.776       

                ! Reaction (315) HOCH2CH2O2 = HOCH2CHO + HO2                                        
                RC(:,315) = 2.00D-12*RO2(:)*0.224       

                ! Reaction (316) RN9O2 = CH3CHO + HCHO + HO2                                        
                RC(:,316) = 8.80D-13*RO2(:)                 

                ! Reaction (317) RN12O2 = CH3CHO + CH3CHO + HO2                                     
                RC(:,317) = 8.80D-13*RO2(:)                 

                ! Reaction (318) RN15O2 = C2H5CHO + CH3CHO + HO2                                    
                RC(:,318) = 8.80D-13*RO2(:)                 

                ! Reaction (319) RN18O2 = C2H5CHO + C2H5CHO + HO2                                   
                RC(:,319) = 8.80D-13*RO2(:)                 

                ! Reaction (320) RN15AO2 = CARB13 + HO2                                             
                RC(:,320) = 8.80D-13*RO2(:)                 

                ! Reaction (321) RN18AO2 = CARB16 + HO2                                             
                RC(:,321) = 8.80D-13*RO2(:)                 

                ! Reaction (322) CH3CO3 = CH3O2                                                     
                RC(:,322) = 1.00D-11*RO2(:)                 

                ! Reaction (323) C2H5CO3 = C2H5O2                                                   
                RC(:,323) = 1.00D-11*RO2(:)                 

                ! Reaction (324) HOCH2CO3 = HCHO + HO2                                              
                RC(:,324) = 1.00D-11*RO2(:)                 

                ! Reaction (325) RN8O2 = CH3CO3 + HCHO                                              
                RC(:,325) = 1.40D-12*RO2(:)                 

                ! Reaction (326) RN11O2 = CH3CO3 + CH3CHO                                           
                RC(:,326) = 1.40D-12*RO2(:)                 

                ! Reaction (327) RN14O2 = C2H5CO3 + CH3CHO                                          
                RC(:,327) = 1.40D-12*RO2(:)                 

                ! Reaction (328) RN17O2 = RN16AO2                                                   
                RC(:,328) = 1.40D-12*RO2(:)                 

                ! Reaction (329) RU14O2 = UCARB12 + HO2                                             
                RC(:,329) = 1.26D-12*RO2(:)*0.1        

                ! Reaction (330) RU14O2 = UCARB10 + HCHO + HO2                                      
                RC(:,330) = 1.26D-12*RO2(:)*0.9       

                ! Reaction (331) RU12O2 = CH3CO3 + HOCH2CHO                                         
                RC(:,331) = 4.20D-13*RO2(:)*0.7           

                ! Reaction (332) RU12O2 = CARB7 + HOCH2CHO + HO2                                    
                RC(:,332) = 4.20D-13*RO2(:)*0.3            

                ! Reaction (333) RU10O2 = CH3CO3 + HOCH2CHO                                         
                RC(:,333) = 1.83D-12*RO2(:)*0.7            

                ! Reaction (334) RU10O2 = CARB6 + HCHO + HO2                                        
                RC(:,334) = 1.83D-12*RO2(:)*0.3            

                ! Reaction (335) RU10O2 = CARB7 + HCHO + HO2                                        
                RC(:,335) = 1.83D-12*RO2(:)*0.0             

                ! Reaction (336) NRN6O2 = HCHO + HCHO + NO2                                         
                RC(:,336) = 6.00D-13*RO2(:)                 

                ! Reaction (337) NRN9O2 = CH3CHO + HCHO + NO2                                       
                RC(:,337) = 2.30D-13*RO2(:)                 

                ! Reaction (338) NRN12O2 = CH3CHO + CH3CHO + NO2                                    
                RC(:,338) = 2.50D-13*RO2(:)                 

                ! Reaction (339) NRU14O2 = NUCARB12 + HO2                                           
                RC(:,339) = 1.30D-12*RO2(:)                 

                ! Reaction (340) NRU12O2 = NOA + CO + HO2                                           
                RC(:,340) = 9.60D-13*RO2(:)                 

                ! Reaction (341) RTN28O2 = TNCARB26 + HO2                                           
                RC(:,341) = 2.85D-13*RO2(:)                 

                ! Reaction (342) NRTN28O2 = TNCARB26 + NO2                                          
                RC(:,342) = 1.00D-13*RO2(:)                 

                ! Reaction (343) RTN26O2 = RTN25O2                                                  
                RC(:,343) = 1.00D-11*RO2(:)                   

                ! Reaction (344) RTN25O2 = RTN24O2                                                  
                RC(:,344) = 1.30D-12*RO2(:)           

                ! Reaction (345) RTN24O2 = RTN23O2                                                  
                RC(:,345) = 6.70D-15*RO2(:)             

                ! Reaction (346) RTN23O2 = CH3COCH3 + RTN14O2                                       
                RC(:,346) = 6.70D-15*RO2(:)            

                ! Reaction (347) RTN14O2 = HCHO + TNCARB10 + HO2                                    
                RC(:,347) = 8.80D-13*RO2(:)        

                ! Reaction (348) RTN10O2 = RN8O2 + CO                                               
                RC(:,348) = 2.00D-12*RO2(:)        

                ! Reaction (349) RTX28O2 = TXCARB24 + HCHO + HO2                                    
                RC(:,349) = 2.00D-12*RO2(:)       

                ! Reaction (350) RTX24O2 = TXCARB22 + HO2                                           
                RC(:,350) = 2.50D-13*RO2(:)       

                ! Reaction (351) RTX22O2 = CH3COCH3 + RN13O2                                        
                RC(:,351) = 2.50D-13*RO2(:)       

                ! Reaction (352) NRTX28O2 = TXCARB24 + HCHO + NO2                                   
                RC(:,352) = 9.20D-14*RO2(:)       

                ! Reaction (353) OH + CARB14 = RN14O2                                               
                RC(:,353) = 1.87D-11       

                ! Reaction (354) OH + CARB17 = RN17O2                                               
                RC(:,354) = 4.36D-12       

                ! Reaction (355) OH + CARB11A = RN11O2                                              
                RC(:,355) = 3.24D-18*TEMP(:)**2*EXP(414/TEMP(:))

                ! Reaction (356) OH + CARB7 = CARB6 + HO2                                           
                RC(:,356) = 1.60D-12*EXP(305/TEMP(:))      

                ! Reaction (357) OH + CARB10 = CARB9 + HO2                                          
                RC(:,357) = 5.86D-12       

                ! Reaction (358) OH + CARB13 = RN13O2                                               
                RC(:,358) = 1.65D-11       

                ! Reaction (359) OH + CARB16 = RN16O2                                               
                RC(:,359) = 1.25D-11       

                ! Reaction (360) OH + UCARB10 = RU10O2                                              
                RC(:,360) = 3.84D-12*EXP(533/TEMP(:))*0.693     

                ! Reaction (361) NO3 + UCARB10 = RU10O2 + HNO3                                      
                RC(:,361) = KNO3AL(:)*0.415    

                ! Reaction (362) O3 + UCARB10 = HCHO + CH3CO3 + CO + OH                             
                RC(:,362) = 1.20D-15*EXP(-1710/TEMP)*0.32       

                ! Reaction (363) O3 + UCARB10 = HCHO + CARB6 + H2O2                                 
                RC(:,363) = 1.20D-15*EXP(-1710/TEMP)*0.68       

                ! Reaction (364) OH + HOCH2CHO = HOCH2CO3                                           
                RC(:,364) = 1.00D-11       

                ! Reaction (365) NO3 + HOCH2CHO = HOCH2CO3 + HNO3                                   
                RC(:,365) = KNO3AL(:)        

                ! Reaction (366) OH + CARB3 = CO + CO + HO2                                         
                RC(:,366) = 3.10D-12*EXP(340/TEMP(:))*0.8       

                ! Reaction (367) OH + CARB6 = CH3CO3 + CO                                           
                RC(:,367) = 1.90D-12*EXP(575/TEMP(:))      

                ! Reaction (368) OH + CARB9 = RN9O2                                                 
                RC(:,368) = 2.40D-13       

                ! Reaction (369) OH + CARB12 = RN12O2                                               
                RC(:,369) = 1.38D-12       

                ! Reaction (370) OH + CARB15 = RN15O2                                               
                RC(:,370) = 4.81D-12       

                ! Reaction (371) OH + CCARB12 = RN12O2                                              
                RC(:,371) = 4.79D-12       

                ! Reaction (372) OH + UCARB12 = RU12O2                                              
                RC(:,372) = 4.52D-11            

                ! Reaction (373) NO3 + UCARB12 = RU12O2 + HNO3                                      
                RC(:,373) = KNO3AL(:)*4.25    

                ! Reaction (374) O3 + UCARB12 = HOCH2CHO + CH3CO3 + CO + OH                         
                RC(:,374) = 2.40D-17*0.89   

                ! Reaction (375) O3 + UCARB12 = HOCH2CHO + CARB6 + H2O2                             
                RC(:,375) = 2.40D-17*0.11   

                ! Reaction (376) OH + NUCARB12 = NRU12O2                                            
                RC(:,376) = 4.16D-11            

                ! Reaction (377) OH + NOA = CARB6 + NO2                                             
                RC(:,377) = 6.70D-13             

                ! Reaction (378) OH + UDCARB8 = C2H5O2                                              
                RC(:,378) = 5.20D-11*0.50        

                ! Reaction (379) OH + UDCARB8 = ANHY + HO2                                          
                RC(:,379) = 5.20D-11*0.50        

                ! Reaction (380) OH + UDCARB11 = RN10O2                                             
                RC(:,380) = 5.58D-11*0.55     

                ! Reaction (381) OH + UDCARB11 = ANHY + CH3O2                                       
                RC(:,381) = 5.58D-11*0.45     

                ! Reaction (382) OH + UDCARB14 = RN13O2                                             
                RC(:,382) = 7.00D-11*0.55     

                ! Reaction (383) OH + UDCARB14 = ANHY + C2H5O2                                      
                RC(:,383) = 7.00D-11*0.45     

                ! Reaction (384) OH + TNCARB26 = RTN26O2                                            
                RC(:,384) = 4.20D-11           

                ! Reaction (385) OH + TNCARB15 = RN15AO2                                            
                RC(:,385) = 1.00D-12           

                ! Reaction (386) OH + TNCARB10 = RTN10O2                                            
                RC(:,386) = 1.00D-10           

                ! Reaction (387) NO3 + TNCARB26 = RTN26O2 + HNO3                                    
                RC(:,387) = 3.80D-14            

                ! Reaction (388) NO3 + TNCARB10 = RTN10O2 + HNO3                                    
                RC(:,388) = KNO3AL(:)*5.5      

                ! Reaction (389) OH + RCOOH25 = RTN25O2                                             
                RC(:,389) = 6.65D-12            

                ! Reaction (390) OH + TXCARB24 = RTX24O2                                            
                RC(:,390) = 1.55D-11           

                ! Reaction (391) OH + TXCARB22 = RTX22O2                                            
                RC(:,391) = 4.55D-12           

                ! Reaction (392) OH + CH3NO3 = HCHO + NO2                                           
                RC(:,392) = 1.00D-14*EXP(1060/TEMP(:))      

                ! Reaction (393) OH + C2H5NO3 = CH3CHO + NO2                                        
                RC(:,393) = 4.40D-14*EXP(720/TEMP(:))       

                ! Reaction (394) OH + RN10NO3 = C2H5CHO + NO2                                       
                RC(:,394) = 7.30D-13                     

                ! Reaction (395) OH + IC3H7NO3 = CH3COCH3 + NO2                                     
                RC(:,395) = 4.90D-13                     

                ! Reaction (396) OH + RN13NO3 = CARB11A + NO2                                       
                RC(:,396) = 9.20D-13                     

                ! Reaction (397) OH + RN16NO3 = CARB14 + NO2                                        
                RC(:,397) = 1.85D-12                     

                ! Reaction (398) OH + RN19NO3 = CARB17 + NO2                                        
                RC(:,398) = 3.02D-12                     

                ! Reaction (399) OH + HOC2H4NO3 = HOCH2CHO + NO2                                    
                RC(:,399) = 1.09D-12               

                ! Reaction (400) OH + RN9NO3 = CARB7 + NO2                                          
                RC(:,400) = 1.31D-12               

                ! Reaction (401) OH + RN12NO3 = CARB10 + NO2                                        
                RC(:,401) = 1.79D-12               

                ! Reaction (402) OH + RN15NO3 = CARB13 + NO2                                        
                RC(:,402) = 1.03D-11               

                ! Reaction (403) OH + RN18NO3 = CARB16 + NO2                                        
                RC(:,403) = 1.34D-11               

                ! Reaction (404) OH + RU14NO3 = UCARB12 + NO2                                       
                RC(:,404) = 3.00D-11*0.34               

                ! Reaction (405) OH + RA13NO3 = CARB3 + UDCARB8 + NO2                               
                RC(:,405) = 7.30D-11               

                ! Reaction (406) OH + RA16NO3 = CARB3 + UDCARB11 + NO2                              
                RC(:,406) = 7.16D-11               

                ! Reaction (407) OH + RA19NO3 = CARB6 + UDCARB11 + NO2                              
                RC(:,407) = 8.31D-11               

                ! Reaction (408) OH + RTN28NO3 = TNCARB26 + NO2                                     
                RC(:,408) = 4.35D-12               

                ! Reaction (409) OH + RTN25NO3 = CH3COCH3 + TNCARB15 + NO2                          
                RC(:,409) = 2.88D-12               

                ! Reaction (410) OH + RTX28NO3 = TXCARB24 + HCHO + NO2                              
                RC(:,410) = 3.53D-12                  

                ! Reaction (411) OH + RTX24NO3 = TXCARB22 + NO2                                     
                RC(:,411) = 6.48D-12                  

                ! Reaction (412) OH + RTX22NO3 = CH3COCH3 + CCARB12 + NO2                           
                RC(:,412) = 4.74D-12                  

                ! Reaction (413) OH + AROH14 = RAROH14                                              
                RC(:,413) = 2.63D-11             

                ! Reaction (414) NO3 + AROH14 = RAROH14 + HNO3                                      
                RC(:,414) = 3.78D-12               

                ! Reaction (415) RAROH14 + NO2 = ARNOH14                                            
                RC(:,415) = 2.08D-12               

                ! Reaction (416) OH + ARNOH14 = CARB13 + NO2                                        
                RC(:,416) = 9.00D-13               

                ! Reaction (417) NO3 + ARNOH14 = CARB13 + NO2 + HNO3                                
                RC(:,417) = 9.00D-14               

                ! Reaction (418) OH + AROH17 = RAROH17                                              
                RC(:,418) = 4.65D-11               

                ! Reaction (419) NO3 + AROH17 = RAROH17 + HNO3                                      
                RC(:,419) = 1.25D-11               

                ! Reaction (420) RAROH17 + NO2 = ARNOH17                                            
                RC(:,420) = 2.08D-12               

                ! Reaction (421) OH + ARNOH17 = CARB16 + NO2                                        
                RC(:,421) = 1.53D-12               

                ! Reaction (422) NO3 + ARNOH17 = CARB16 + NO2 + HNO3                                
                RC(:,422) = 3.13D-13               

                ! Reaction (423) OH + CH3OOH = CH3O2                                                
                RC(:,423) = 1.90D-11*EXP(190/TEMP(:))       

                ! Reaction (424) OH + CH3OOH = HCHO + OH                                            
                RC(:,424) = 1.00D-11*EXP(190/TEMP(:))       

                ! Reaction (425) OH + C2H5OOH = CH3CHO + OH                                         
                RC(:,425) = 1.36D-11               

                ! Reaction (426) OH + RN10OOH = C2H5CHO + OH                                        
                RC(:,426) = 1.89D-11               

                ! Reaction (427) OH + IC3H7OOH = CH3COCH3 + OH                                      
                RC(:,427) = 2.78D-11               

                ! Reaction (428) OH + RN13OOH = CARB11A + OH                                        
                RC(:,428) = 3.57D-11               

                ! Reaction (429) OH + RN16OOH = CARB14 + OH                                         
                RC(:,429) = 4.21D-11               

                ! Reaction (430) OH + RN19OOH = CARB17 + OH                                         
                RC(:,430) = 4.71D-11               

                ! Reaction (431) OH + CH3CO3H = CH3CO3                                              
                RC(:,431) = 3.70D-12                     

                ! Reaction (432) OH + C2H5CO3H = C2H5CO3                                            
                RC(:,432) = 4.42D-12                     

                ! Reaction (433) OH + HOCH2CO3H = HOCH2CO3                                          
                RC(:,433) = 6.19D-12                     

                ! Reaction (434) OH + RN8OOH = CARB6 + OH                                           
                RC(:,434) = 1.2D-11                     

                ! Reaction (435) OH + RN11OOH = CARB9 + OH                                          
                RC(:,435) = 2.50D-11                     

                ! Reaction (436) OH + RN14OOH = CARB12 + OH                                         
                RC(:,436) = 3.20D-11                     

                ! Reaction (437) OH + RN17OOH = CARB15 + OH                                         
                RC(:,437) = 3.35D-11                     

                ! Reaction (438) OH + RU14OOH = UCARB12 + OH                                        
                RC(:,438) = 7.51D-11                     

                ! Reaction (439) OH + RU12OOH = RU12O2                                              
                RC(:,439) = 3.50D-11                     

                ! Reaction (440) OH + RU10OOH = RU10O2                                              
                RC(:,440) = 3.50D-11                     

                ! Reaction (441) OH + NRU14OOH = NUCARB12 + OH                                      
                RC(:,441) = 1.03D-10                     

                ! Reaction (442) OH + NRU12OOH = NOA + CO + OH                                      
                RC(:,442) = 2.65D-11                     

                ! Reaction (443) OH + HOC2H4OOH = HOCH2CHO + OH                                     
                RC(:,443) = 2.13D-11               

                ! Reaction (444) OH + RN9OOH = CARB7 + OH                                           
                RC(:,444) = 2.50D-11               

                ! Reaction (445) OH + RN12OOH = CARB10 + OH                                         
                RC(:,445) = 3.25D-11               

                ! Reaction (446) OH + RN15OOH = CARB13 + OH                                         
                RC(:,446) = 3.74D-11               

                ! Reaction (447) OH + RN18OOH = CARB16 + OH                                         
                RC(:,447) = 3.83D-11               

                ! Reaction (448) OH + NRN6OOH = HCHO + HCHO + NO2 + OH                              
                RC(:,448) = 5.22D-12               

                ! Reaction (449) OH + NRN9OOH = CH3CHO + HCHO + NO2 + OH                            
                RC(:,449) = 6.50D-12               

                ! Reaction (450) OH + NRN12OOH = CH3CHO + CH3CHO + NO2 + OH                         
                RC(:,450) = 7.15D-12               

                ! Reaction (451) OH + RA13OOH = CARB3 + UDCARB8 + OH                                
                RC(:,451) = 9.77D-11               

                ! Reaction (452) OH + RA16OOH = CARB3 + UDCARB11 + OH                               
                RC(:,452) = 9.64D-11               

                ! Reaction (453) OH + RA19OOH = CARB6 + UDCARB11 + OH                               
                RC(:,453) = 1.12D-10               

                ! Reaction (454) OH + RTN28OOH = TNCARB26 + OH                                      
                RC(:,454) = 2.38D-11               

                ! Reaction (455) OH + RTN26OOH = RTN26O2                                            
                RC(:,455) = 1.20D-11               

                ! Reaction (456) OH + NRTN28OOH = TNCARB26 + NO2 + OH                               
                RC(:,456) = 9.50D-12               

                ! Reaction (457) OH + RTN25OOH = RTN25O2                                            
                RC(:,457) = 1.66D-11               

                ! Reaction (458) OH + RTN24OOH = RTN24O2                                            
                RC(:,458) = 1.05D-11               

                ! Reaction (459) OH + RTN23OOH = RTN23O2                                            
                RC(:,459) = 2.05D-11               

                ! Reaction (460) OH + RTN14OOH = RTN14O2                                            
                RC(:,460) = 8.69D-11               

                ! Reaction (461) OH + RTN10OOH = RTN10O2                                            
                RC(:,461) = 4.23D-12               

                ! Reaction (462) OH + RTX28OOH = RTX28O2                                            
                RC(:,462) = 2.00D-11               

                ! Reaction (463) OH + RTX24OOH = TXCARB22 + OH                                      
                RC(:,463) = 8.59D-11               

                ! Reaction (464) OH + RTX22OOH = CH3COCH3 + CCARB12 + OH                            
                RC(:,464) = 7.50D-11               

                ! Reaction (465) OH + NRTX28OOH = NRTX28O2                                          
                RC(:,465) = 9.58D-12               

                ! Reaction (466) OH + ANHY = HOCH2CH2O2                                             
                RC(:,466) = 1.50D-12        

                ! Reaction (467) CH3CO3 + NO2 = PAN                                                 
                RC(:,467) = KFPAN(:)                        

                ! Reaction (468) PAN = CH3CO3 + NO2                                                 
                RC(:,468) = KBPAN(:)                        

                ! Reaction (469) C2H5CO3 + NO2 = PPN                                                
                RC(:,469) = KFPAN(:)                        

                ! Reaction (470) PPN = C2H5CO3 + NO2                                                
                RC(:,470) = KBPAN(:)                        

                ! Reaction (471) HOCH2CO3 + NO2 = PHAN                                              
                RC(:,471) = 1.125D-11*(TEMP(:)/300)**(-1.105)                            

                ! Reaction (472) PHAN = HOCH2CO3 + NO2                                              
                RC(:,472) = 5.2D16*EXP(-13850/TEMP(:))                       

                ! Reaction (473) OH + PAN = HCHO + CO + NO2                                         
                RC(:,473) = 3.00D-14    

                ! Reaction (474) OH + PPN = CH3CHO + CO + NO2                                       
                RC(:,474) = 1.27D-12                       

                ! Reaction (475) OH + PHAN = HCHO + CO + NO2                                        
                RC(:,475) = 1.12D-12                       

                ! Reaction (476) RU12O2 + NO2 = RU12PAN                                             
                RC(:,476) = KFPAN(:)*0.061             

                ! Reaction (477) RU12PAN = RU12O2 + NO2                                             
                RC(:,477) = KBPAN(:)                   

                ! Reaction (478) RU10O2 + NO2 = MPAN                                                
                RC(:,478) = KFPAN(:)*0.041             

                ! Reaction (479) MPAN = RU10O2 + NO2                                                
                RC(:,479) = KBPAN(:)                  

                ! Reaction (480) OH + MPAN = CARB7 + CO + NO2                                       
                RC(:,480) = 2.90D-11*0.22 

                ! Reaction (481) OH + RU12PAN = UCARB10 + NO2                                       
                RC(:,481) = 2.52D-11 

                ! Reaction (482) RTN26O2 + NO2 = RTN26PAN                                           
                RC(:,482) = 1.125D-11*(TEMP(:)/300)**(-1.105)*0.722      

                ! Reaction (483) RTN26PAN = RTN26O2 + NO2                                           
                RC(:,483) = 5.2D16*EXP(-13850/TEMP(:))                   

                ! Reaction (484) OH + RTN26PAN = CH3COCH3 + CARB16 + NO2                            
                RC(:,484) = 3.66D-12  

                ! Reaction (485) RTN28NO3 = P2604                                                   
                RC(:,485) = KIN(:)  		

                ! Reaction (486) P2604 = RTN28NO3                                                   
                RC(:,486) = KOUT2604(:)

                ! Reaction (487) RTX28NO3 = P4608                                                   
                RC(:,487) = KIN(:)	

                ! Reaction (488) P4608 = RTX28NO3                                                   
                RC(:,488) = KOUT4608(:) 	

                ! Reaction (489) RCOOH25 = P2631                                                    
                RC(:,489) = KIN(:)  		

                ! Reaction (490) P2631 = RCOOH25                                                    
                RC(:,490) = KOUT2631(:) 	

                ! Reaction (491) RTN24OOH = P2635                                                   
                RC(:,491) = KIN(:)  		

                ! Reaction (492) P2635 = RTN24OOH                                                   
                RC(:,492) = KOUT2635(:) 	

                ! Reaction (493) RTX28OOH = P4610                                                   
                RC(:,493) = KIN(:)  		

                ! Reaction (494) P4610 = RTX28OOH                                                   
                RC(:,494) = KOUT4610(:) 	

                ! Reaction (495) RTN28OOH = P2605                                                   
                RC(:,495) = KIN(:)  		

                ! Reaction (496) P2605 = RTN28OOH                                                   
                RC(:,496) = KOUT2605(:) 	

                ! Reaction (497) RTN26OOH = P2630                                                   
                RC(:,497) = KIN(:)		

                ! Reaction (498) P2630 = RTN26OOH                                                   
                RC(:,498) = KOUT2630(:) 	

                ! Reaction (499) RTN26PAN = P2629                                                   
                RC(:,499) = KIN(:) 		

                ! Reaction (500) P2629 = RTN26PAN                                                   
                RC(:,500) = KOUT2629(:) 	

                ! Reaction (501) RTN25OOH = P2632                                                   
                RC(:,501) = KIN(:) 		

                ! Reaction (502) P2632 = RTN25OOH                                                   
                RC(:,502) = KOUT2632(:) 	

                ! Reaction (503) RTN23OOH = P2637                                                   
                RC(:,503) = KIN(:)  		

                ! Reaction (504) P2637 = RTN23OOH                                                   
                RC(:,504) = KOUT2637(:) 	

                ! Reaction (505) ARNOH14 = P3612                                                    
                RC(:,505) = KIN(:)  		

                ! Reaction (506) P3612 = ARNOH14                                                    
                RC(:,506) = KOUT3612(:) 	

                ! Reaction (507) ARNOH17 = P3613                                                    
                RC(:,507) = KIN(:) 		

                ! Reaction (508) P3613 = ARNOH17                                                    
                RC(:,508) = KOUT3613(:) 	

                ! Reaction (509) ANHY = P3442                                                       
                RC(:,509) = KIN(:)  		

                ! Reaction (510) P3442 = ANHY                                                       
                RC(:,510) = KOUT3442(:)

    END SUBROUTINE CHEMCO

    SUBROUTINE CALC_J(NCELL, SZA, J)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NCELL
        DOUBLE PRECISION, INTENT(IN) :: SZA(NCELL)
        DOUBLE PRECISION, INTENT(OUT) :: J(NCELL,57)
        INTEGER :: CELL
        DOUBLE PRECISION :: PI, COSX, SECX

        PI = 4.00D+00*ATAN(1.00D+00)

        DO CELL = 1,NCELL
            IF((SZA(CELL) - 0.5*PI).LT.0) THEN
                COSX = COS(SZA(CELL))
                SECX = 1.0/COSX
                J(CELL,1)=6.073D-05*(COSX**(1.743))*EXP(-0.474*SECX) 
                J(CELL,2)=4.775D-04*(COSX**(0.298))*EXP(-0.080*SECX) 
                J(CELL,3)=1.041D-05*(COSX**(0.723))*EXP(-0.279*SECX) 
                J(CELL,4)=1.165D-02*(COSX**(0.244))*EXP(-0.267*SECX) 
                J(CELL,5)=2.485D-02*(COSX**(0.168))*EXP(-0.108*SECX) 
                J(CELL,6)=1.747D-01*(COSX**(0.155))*EXP(-0.125*SECX) 
                J(CELL,7)=2.644D-03*(COSX**(0.261))*EXP(-0.288*SECX) 
                J(CELL,8)=9.312D-07*(COSX**(1.230))*EXP(-0.307*SECX) 
                J(CELL,11)=4.642D-05*(COSX**(0.762))*EXP(-0.353*SECX)
                J(CELL,12)=6.853D-05*(COSX**(0.477))*EXP(-0.323*SECX) 
                J(CELL,13)=7.344D-06*(COSX**(1.202))*EXP(-0.417*SECX) 
                J(CELL,14)=2.879D-05*(COSX**(1.067))*EXP(-0.358*SECX) 
                J(CELL,15)=2.792D-05*(COSX**(0.805))*EXP(-0.338*SECX) 
                J(CELL,16)=1.675D-05*(COSX**(0.805))*EXP(-0.338*SECX) 
                J(CELL,17)=7.914D-05*(COSX**(0.764))*EXP(-0.364*SECX) 
                J(CELL,18)=1.140D-05*(COSX**(0.396))*EXP(-0.298*SECX) 
                J(CELL,19)=1.140D-05*(COSX**(0.396))*EXP(-0.298*SECX) 
                J(CELL,21)=7.992D-07*(COSX**(1.578))*EXP(-0.271*SECX) 
                J(CELL,22)=5.804D-06*(COSX**(1.092))*EXP(-0.377*SECX) 
                J(CELL,23)=1.836D-05*(COSX**(0.395))*EXP(-0.296*SECX) 
                J(CELL,24)=1.836D-05*(COSX**(0.395))*EXP(-0.296*SECX) 
                J(CELL,31)=6.845D-05*(COSX**(0.130))*EXP(-0.201*SECX) 
                J(CELL,32)=1.032D-05*(COSX**(0.130))*EXP(-0.201*SECX) 
                J(CELL,33)=3.802D-05*(COSX**(0.644))*EXP(-0.312*SECX) 
                J(CELL,34)=1.537D-04*(COSX**(0.170))*EXP(-0.208*SECX) 
                J(CELL,35)=3.326D-04*(COSX**(0.148))*EXP(-0.215*SECX) 
                J(CELL,41)=7.649D-06*(COSX**(0.682))*EXP(-0.279*SECX) 
                J(CELL,51)=1.588D-06*(COSX**(1.154))*EXP(-0.318*SECX) 
                J(CELL,52)=1.907D-06*(COSX**(1.244))*EXP(-0.335*SECX) 
                J(CELL,53)=2.485D-06*(COSX**(1.196))*EXP(-0.328*SECX) 
                J(CELL,54)=4.095D-06*(COSX**(1.111))*EXP(-0.316*SECX) 
                J(CELL,55)=1.135D-05*(COSX**(0.974))*EXP(-0.309*SECX) 
                J(CELL,56)=7.549D-06*(COSX**(1.015))*EXP(-0.324*SECX) 
                J(CELL,57)=3.363D-06*(COSX**(1.296))*EXP(-0.322*SECX)
            ELSE
                J(CELL,:) = 1.0E-30
            END IF
            
        END DO

    END SUBROUTINE CALC_J

    SUBROUTINE PHOTOL(NCELL, J, DJ)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NCELL
        DOUBLE PRECISION, INTENT(IN) :: J(NCELL,57)
        DOUBLE PRECISION, INTENT(OUT) :: DJ(NCELL,96)
        ! Photol Reaction (1) O3 = O1D                                                           
        DJ(:,1) = J(:,1)                             

        ! Photol Reaction (2) O3 = O                                                             
        DJ(:,2) = J(:,2)                             

        ! Photol Reaction (3) H2O2 = OH + OH                                                     
        DJ(:,3) = J(:,3)                             

        ! Photol Reaction (4) NO2 = NO + O                                                       
        DJ(:,4) = J(:,4)                             

        ! Photol Reaction (5) NO3 = NO                                                           
        DJ(:,5) = J(:,5)                             

        ! Photol Reaction (6) NO3 = NO2 + O                                                      
        DJ(:,6) = J(:,6)                             

        ! Photol Reaction (7) HONO = OH + NO                                                     
        DJ(:,7) = J(:,7)                             

        ! Photol Reaction (8) HNO3 = OH + NO2                                                    
        DJ(:,8) = J(:,8)                             

        ! Photol Reaction (9) HCHO = CO + HO2 + HO2                                              
        DJ(:,9) = J(:,11)                        

        ! Photol Reaction (10) HCHO = H2 + CO                                                     
        DJ(:,10) = J(:,12)                        

        ! Photol Reaction (11) CH3CHO = CH3O2 + HO2 + CO                                          
        DJ(:,11) = J(:,13)                        

        ! Photol Reaction (12) C2H5CHO = C2H5O2 + CO + HO2                                        
        DJ(:,12) = J(:,14)                        

        ! Photol Reaction (13) CH3COCH3 = CH3CO3 + CH3O2                                          
        DJ(:,13) = J(:,21)                        

        ! Photol Reaction (14) MEK = CH3CO3 + C2H5O2                                              
        DJ(:,14) = J(:,22)                        

        ! Photol Reaction (15) CARB14 = CH3CO3 + RN10O2                                           
        DJ(:,15) = J(:,22)*4.74               

        ! Photol Reaction (16) CARB17 = RN8O2 + RN10O2                                            
        DJ(:,16) = J(:,22)*1.33               

        ! Photol Reaction (17) CARB11A = CH3CO3 + C2H5O2                                          
        DJ(:,17) = J(:,22)                        

        ! Photol Reaction (18) CARB7 = CH3CO3 + HCHO + HO2                                        
        DJ(:,18) = J(:,22)                        

        ! Photol Reaction (19) CARB10 = CH3CO3 + CH3CHO + HO2                                     
        DJ(:,19) = J(:,22)                        

        ! Photol Reaction (20) CARB13 = RN8O2 + CH3CHO + HO2                                      
        DJ(:,20) = J(:,22)*3.00               

        ! Photol Reaction (21) CARB16 = RN8O2 + C2H5CHO + HO2                                     
        DJ(:,21) = J(:,22)*3.35               

        ! Photol Reaction (22) HOCH2CHO = HCHO + CO + HO2 + HO2                                   
        DJ(:,22) = J(:,15)                        

        ! Photol Reaction (23) UCARB10 = CH3CO3 + HCHO + HO2                                      
        DJ(:,23) = J(:,18)*2                       

        ! Photol Reaction (24) CARB3 = CO + CO + HO2 + HO2                                        
        DJ(:,24) = J(:,33)                        

        ! Photol Reaction (25) CARB6 = CH3CO3 + CO + HO2                                          
        DJ(:,25) = J(:,34)                        

        ! Photol Reaction (26) CARB9 = CH3CO3 + CH3CO3                                            
        DJ(:,26) = J(:,35)                        

        ! Photol Reaction (27) CARB12 = CH3CO3 + RN8O2                                            
        DJ(:,27) = J(:,35)                        

        ! Photol Reaction (28) CARB15 = RN8O2 + RN8O2                                             
        DJ(:,28) = J(:,35)                        

        ! Photol Reaction (29) UCARB12 = CH3CO3 + HOCH2CHO + CO + HO2                             
        DJ(:,29) = J(:,18)*2           

        ! Photol Reaction (30) NUCARB12 = NOA + CO + CO + HO2 + HO2                               
        DJ(:,30) = J(:,18)             

        ! Photol Reaction (31) NOA = CH3CO3 + HCHO + NO2                                          
        DJ(:,31) = J(:,56)             

        ! Photol Reaction (32) NOA = CH3CO3 + HCHO + NO2                                          
        DJ(:,32) = J(:,57)             

        ! Photol Reaction (33) UDCARB8 = C2H5O2 + HO2                                             
        DJ(:,33) = J(:,4)*0.02*0.64   

        ! Photol Reaction (34) UDCARB8 = ANHY + HO2 + HO2                                         
        DJ(:,34) = J(:,4)*0.02*0.36   

        ! Photol Reaction (35) UDCARB11 = RN10O2 + HO2                                            
        DJ(:,35) = J(:,4)*0.02*0.55   

        ! Photol Reaction (36) UDCARB11 = ANHY + HO2 + CH3O2                                      
        DJ(:,36) = J(:,4)*0.02*0.45   

        ! Photol Reaction (37) UDCARB14 = RN13O2 + HO2                                            
        DJ(:,37) = J(:,4)*0.02*0.55   

        ! Photol Reaction (38) UDCARB14 = ANHY + HO2 + C2H5O2                                     
        DJ(:,38) = J(:,4)*0.02*0.45   

        ! Photol Reaction (39) TNCARB26 = RTN26O2 + HO2                                           
        DJ(:,39) = J(:,15)             

        ! Photol Reaction (40) TNCARB10 = CH3CO3 + CH3CO3 + CO                                    
        DJ(:,40) = J(:,35)*0.5        

        ! Photol Reaction (41) CH3NO3 = HCHO + HO2 + NO2                                          
        DJ(:,41) = J(:,51)                        

        ! Photol Reaction (42) C2H5NO3 = CH3CHO + HO2 + NO2                                       
        DJ(:,42) = J(:,52)                        

        ! Photol Reaction (43) RN10NO3 = C2H5CHO + HO2 + NO2                                      
        DJ(:,43) = J(:,53)                        

        ! Photol Reaction (44) IC3H7NO3 = CH3COCH3 + HO2 + NO2                                    
        DJ(:,44) = J(:,54)                        

        ! Photol Reaction (45) RN13NO3 =  CH3CHO + C2H5O2 + NO2                                   
        DJ(:,45) = J(:,53)*BR01(:)               

        ! Photol Reaction (46) RN13NO3 =  CARB11A + HO2 + NO2                                     
        DJ(:,46) = J(:,53)*(1-BR01(:))           

        ! Photol Reaction (47) RN16NO3 = RN15O2 + NO2                                             
        DJ(:,47) = J(:,53)                        

        ! Photol Reaction (48) RN19NO3 = RN18O2 + NO2                                             
        DJ(:,48) = J(:,53)                        

        ! Photol Reaction (49) RA13NO3 = CARB3 + UDCARB8 + HO2 + NO2                              
        DJ(:,49) = J(:,54)                    

        ! Photol Reaction (50) RA16NO3 = CARB3 + UDCARB11 + HO2 + NO2                             
        DJ(:,50) = J(:,54)                    

        ! Photol Reaction (51) RA19NO3 = CARB6 + UDCARB11 + HO2 + NO2                             
        DJ(:,51) = J(:,54)                    

        ! Photol Reaction (52) RTX24NO3 = TXCARB22 + HO2 + NO2                                    
        DJ(:,52) = J(:,54)                    

        ! Photol Reaction (53) CH3OOH = HCHO + HO2 + OH                                           
        DJ(:,53) = J(:,41)                        

        ! Photol Reaction (54) C2H5OOH = CH3CHO + HO2 + OH                                        
        DJ(:,54) = J(:,41)                        

        ! Photol Reaction (55) RN10OOH = C2H5CHO + HO2 + OH                                       
        DJ(:,55) = J(:,41)                        

        ! Photol Reaction (56) IC3H7OOH = CH3COCH3 + HO2 + OH                                     
        DJ(:,56) = J(:,41)                        

        ! Photol Reaction (57) RN13OOH =  CH3CHO + C2H5O2 + OH                                    
        DJ(:,57) = J(:,41)*BR01(:)         

        ! Photol Reaction (58) RN13OOH =  CARB11A + HO2 + OH                                      
        DJ(:,58) = J(:,41)*(1-BR01(:))     

        ! Photol Reaction (59) RN16OOH = RN15AO2 + OH                                             
        DJ(:,59) = J(:,41)                        

        ! Photol Reaction (60) RN19OOH = RN18AO2 + OH                                             
        DJ(:,60) = J(:,41)                        

        ! Photol Reaction (61) CH3CO3H = CH3O2 + OH                                               
        DJ(:,61) = J(:,41)                        

        ! Photol Reaction (62) C2H5CO3H = C2H5O2 + OH                                             
        DJ(:,62) = J(:,41)                        

        ! Photol Reaction (63) HOCH2CO3H = HCHO + HO2 + OH                                        
        DJ(:,63) = J(:,41)                        

        ! Photol Reaction (64) RN8OOH = C2H5O2 + OH                                               
        DJ(:,64) = J(:,41)                        

        ! Photol Reaction (65) RN11OOH = RN10O2 + OH                                              
        DJ(:,65) = J(:,41)                        

        ! Photol Reaction (66) RN14OOH = RN13O2 + OH                                              
        DJ(:,66) = J(:,41)                        

        ! Photol Reaction (67) RN17OOH = RN16O2 + OH                                              
        DJ(:,67) = J(:,41)                        

        ! Photol Reaction (68) RU14OOH = UCARB12 + HO2 + OH                                       
        DJ(:,68) = J(:,41)*0.252              

        ! Photol Reaction (69) RU14OOH = UCARB10 + HCHO + HO2 + OH                                
        DJ(:,69) = J(:,41)*0.748              

        ! Photol Reaction (70) RU12OOH = CARB6 + HOCH2CHO + HO2 + OH                              
        DJ(:,70) = J(:,41)                   

        ! Photol Reaction (71) RU10OOH = CH3CO3 + HOCH2CHO + OH                                   
        DJ(:,71) = J(:,41)                   

        ! Photol Reaction (72) NRU14OOH = NUCARB12 + HO2 + OH                                     
        DJ(:,72) = J(:,41)                   

        ! Photol Reaction (73) NRU12OOH = NOA + CO + HO2 + OH                                     
        DJ(:,73) = J(:,41)                   

        ! Photol Reaction (74) HOC2H4OOH = HCHO + HCHO + HO2 + OH                                 
        DJ(:,74) = J(:,41)                  

        ! Photol Reaction (75) RN9OOH = CH3CHO + HCHO + HO2 + OH                                  
        DJ(:,75) = J(:,41)                  

        ! Photol Reaction (76) RN12OOH = CH3CHO + CH3CHO + HO2 + OH                               
        DJ(:,76) = J(:,41)                  

        ! Photol Reaction (77) RN15OOH = C2H5CHO + CH3CHO + HO2 + OH                              
        DJ(:,77) = J(:,41)                  

        ! Photol Reaction (78) RN18OOH = C2H5CHO + C2H5CHO + HO2 + OH                             
        DJ(:,78) = J(:,41)                 

        ! Photol Reaction (79) NRN6OOH = HCHO + HCHO + NO2 + OH                                   
        DJ(:,79) = J(:,41)                  

        ! Photol Reaction (80) NRN9OOH = CH3CHO + HCHO + NO2 + OH                                 
        DJ(:,80) = J(:,41)                  

        ! Photol Reaction (81) NRN12OOH = CH3CHO + CH3CHO + NO2 + OH                              
        DJ(:,81) = J(:,41)                  

        ! Photol Reaction (82) RA13OOH = CARB3 + UDCARB8 + HO2 + OH                               
        DJ(:,82) = J(:,41)                  

        ! Photol Reaction (83) RA16OOH = CARB3 + UDCARB11 + HO2 + OH                              
        DJ(:,83) = J(:,41)                  

        ! Photol Reaction (84) RA19OOH = CARB6 + UDCARB11 + HO2 + OH                              
        DJ(:,84) = J(:,41)                  

        ! Photol Reaction (85) RTN28OOH = TNCARB26 + HO2 + OH                                     
        DJ(:,85) = J(:,41)                  

        ! Photol Reaction (86) NRTN28OOH = TNCARB26 + NO2 + OH                                    
        DJ(:,86) = J(:,41)                  

        ! Photol Reaction (87) RTN26OOH = RTN25O2 + OH                                            
        DJ(:,87) = J(:,41)             

        ! Photol Reaction (88) RTN25OOH = RTN24O2 + OH                                            
        DJ(:,88) = J(:,41)             

        ! Photol Reaction (89) RTN24OOH = RTN23O2 + OH                                            
        DJ(:,89) = J(:,41)             

        ! Photol Reaction (90) RTN23OOH = CH3COCH3 + RTN14O2 + OH                                 
        DJ(:,90) = J(:,41)             

        ! Photol Reaction (91) RTN14OOH = TNCARB10 + HCHO + HO2 + OH                              
        DJ(:,91) = J(:,41)             

        ! Photol Reaction (92) RTN10OOH = RN8O2 + CO + OH                                         
        DJ(:,92) = J(:,41)             

        ! Photol Reaction (93) RTX28OOH = TXCARB24 + HCHO + HO2 + OH                              
        DJ(:,93) = J(:,41)                  

        ! Photol Reaction (94) RTX24OOH = TXCARB22 + HO2 + OH                                     
        DJ(:,94) = J(:,41)                  

        ! Photol Reaction (95) RTX22OOH = CH3COCH3 + RN13O2 + OH                                  
        DJ(:,95) = J(:,41)                  

        ! Photol Reaction (96) NRTX28OOH = TXCARB24 + HCHO + NO2 + OH                             
        DJ(:,96) = J(:,41)

    END SUBROUTINE PHOTOL

    SUBROUTINE DERIV(Y, DJ, RC, FL, DTS)
        IMPLICIT NONE
        DOUBLE PRECISION, INTENT(INOUT) :: Y(:,:), FL(:,:)
        DOUBLE PRECISION, INTENT(IN)    :: DJ(:,:), RC(:,:)
        INTEGER, INTENT(IN)    :: DTS
        
        ! Local temporary array (holds previous state for integration)
        DOUBLE PRECISION, ALLOCATABLE :: YP(:,:)
            
            
        !         ! Allocate and initialize YP from current Y
                IF (.NOT. ALLOCATED(YP)) ALLOCATE(YP(SIZE(Y,1), SIZE(Y,2)))
                YP(:,:) = Y(:,:)
            
                !          O1D              Y(:,1) 
                P(:) = DJ(:,1) * YP(:,6)
                L(:) = 0.0 + (RC(:,7)) + (RC(:,8)) + (RC(:,16) *  H2O(:)) 
                Y(:,1) = P(:)/L(:)
                
                !          O                Y(:,2)
                P(:) = (DJ(:,6) * Y(:,5)) &
                + (DJ(:,2) * Y(:,6)) + (DJ(:,4) * Y(:,4)) &
                + (RC(:,7) * Y(:,1)) + (RC(:,8) * Y(:,1)) 
                L(:) = 0.0 &
                + (RC(:,36) * Y(:,16)) &
                + (RC(:,4) * Y(:,8)) + (RC(:,5) * Y(:,4)) + (RC(:,6) * Y(:,4)) &
                + (RC(:,1)) + (RC(:,2)) + (RC(:,3) * Y(:,6)) 
                Y(:,2) = P(:)/L(:)
                
                !          OH               Y(:,3) 
                P(:) = (DJ(:,95) * Y(:,184)) + (DJ(:,96) * Y(:,185)) &
                + (DJ(:,93) * Y(:,182)) + (DJ(:,94) * Y(:,183)) &
                + (DJ(:,91) * Y(:,180)) + (DJ(:,92) * Y(:,181)) &
                + (DJ(:,89) * Y(:,178)) + (DJ(:,90) * Y(:,179)) &
                + (DJ(:,87) * Y(:,176)) + (DJ(:,88) * Y(:,177)) &
                + (DJ(:,85) * Y(:,174)) + (DJ(:,86) * Y(:,175)) &
                + (DJ(:,83) * Y(:,152)) + (DJ(:,84) * Y(:,153)) &
                + (DJ(:,81) * Y(:,171)) + (DJ(:,82) * Y(:,151)) &
                + (DJ(:,79) * Y(:,169)) + (DJ(:,80) * Y(:,170)) &
                + (DJ(:,77) * Y(:,157)) + (DJ(:,78) * Y(:,158)) &
                + (DJ(:,75) * Y(:,155)) + (DJ(:,76) * Y(:,156)) &
                + (DJ(:,73) * Y(:,173)) + (DJ(:,74) * Y(:,154)) &
                + (DJ(:,71) * Y(:,168)) + (DJ(:,72) * Y(:,172)) &
                + (DJ(:,69) * Y(:,166)) + (DJ(:,70) * Y(:,167)) &
                + (DJ(:,67) * Y(:,165)) + (DJ(:,68) * Y(:,166)) &
                + (DJ(:,65) * Y(:,163)) + (DJ(:,66) * Y(:,164)) &
                + (DJ(:,63) * Y(:,161)) + (DJ(:,64) * Y(:,162)) &
                + (DJ(:,61) * Y(:,159)) + (DJ(:,62) * Y(:,160)) &
                + (DJ(:,59) * Y(:,149)) + (DJ(:,60) * Y(:,150)) &
                + (DJ(:,57) * Y(:,148)) + (DJ(:,58) * Y(:,148)) &
                + (DJ(:,55) * Y(:,146)) + (DJ(:,56) * Y(:,147)) &
                + (DJ(:,53) * Y(:,144)) + (DJ(:,54) * Y(:,145)) &
                + (DJ(:,7) * Y(:,13)) + (DJ(:,8) * Y(:,14)) &
                + (RC(:,464) * Y(:,3) * Y(:,184)) + (DJ(:,3) * Y(:,12) * 2.00) &
                + (RC(:,456) * Y(:,3) * Y(:,175)) + (RC(:,463) * Y(:,3) * Y(:,183)) &
                + (RC(:,453) * Y(:,3) * Y(:,153)) + (RC(:,454) * Y(:,3) * Y(:,174)) &
                + (RC(:,451) * Y(:,3) * Y(:,151)) + (RC(:,452) * Y(:,3) * Y(:,152)) &
                + (RC(:,449) * Y(:,3) * Y(:,170)) + (RC(:,450) * Y(:,3) * Y(:,171)) &
                + (RC(:,447) * Y(:,3) * Y(:,158)) + (RC(:,448) * Y(:,3) * Y(:,169)) &
                + (RC(:,445) * Y(:,3) * Y(:,156)) + (RC(:,446) * Y(:,3) * Y(:,157)) &
                + (RC(:,443) * Y(:,3) * Y(:,154)) + (RC(:,444) * Y(:,3) * Y(:,155)) &
                + (RC(:,441) * Y(:,3) * Y(:,172)) + (RC(:,442) * Y(:,3) * Y(:,173)) &
                + (RC(:,437) * Y(:,3) * Y(:,165)) + (RC(:,438) * Y(:,3) * Y(:,166)) &
                + (RC(:,435) * Y(:,3) * Y(:,163)) + (RC(:,436) * Y(:,3) * Y(:,164)) &
                + (RC(:,430) * Y(:,3) * Y(:,150)) + (RC(:,434) * Y(:,3) * Y(:,162)) &
                + (RC(:,428) * Y(:,3) * Y(:,148)) + (RC(:,429) * Y(:,3) * Y(:,149)) &
                + (RC(:,426) * Y(:,3) * Y(:,146)) + (RC(:,427) * Y(:,3) * Y(:,147)) &
                + (RC(:,424) * Y(:,3) * Y(:,144)) + (RC(:,425) * Y(:,3) * Y(:,145)) &
                + (RC(:,362) * Y(:,6) * Y(:,46)) + (RC(:,374) * Y(:,6) * Y(:,109)) &
                + (RC(:,70) * Y(:,53) * Y(:,6)) + (RC(:,75) * Y(:,59) * Y(:,3)) &
                + (RC(:,61) * Y(:,6) * Y(:,43)) + (RC(:,65) * Y(:,47) * Y(:,6)) &
                + (RC(:,55) * Y(:,6) * Y(:,32)) + (RC(:,57) * Y(:,6) * Y(:,34)) &
                + (RC(:,33) * Y(:,9) * Y(:,5)) + (RC(:,53) * Y(:,6) * Y(:,30)) &
                + (RC(:,21) * Y(:,9) * Y(:,6)) + (RC(:,29) * Y(:,9) * Y(:,8)) &
                + (RC(:,16) * Y(:,1) *  H2O(:)*2.00) 
                L(:) = 0.0 &
                + (RC(:,481) * Y(:,201)) + (RC(:,484) * Y(:,203)) &
                + (RC(:,474) * Y(:,199)) + (RC(:,475) * Y(:,200)) + (RC(:,480) * Y(:,202)) &
                + (RC(:,465) * Y(:,185)) + (RC(:,466) * Y(:,192)) + (RC(:,473) * Y(:,198)) &
                + (RC(:,462) * Y(:,182)) + (RC(:,463) * Y(:,183)) + (RC(:,464) * Y(:,184)) &
                + (RC(:,459) * Y(:,179)) + (RC(:,460) * Y(:,180)) + (RC(:,461) * Y(:,181)) &
                + (RC(:,456) * Y(:,175)) + (RC(:,457) * Y(:,177)) + (RC(:,458) * Y(:,178)) &
                + (RC(:,453) * Y(:,153)) + (RC(:,454) * Y(:,174)) + (RC(:,455) * Y(:,176)) &
                + (RC(:,450) * Y(:,171)) + (RC(:,451) * Y(:,151)) + (RC(:,452) * Y(:,152)) &
                + (RC(:,447) * Y(:,158)) + (RC(:,448) * Y(:,169)) + (RC(:,449) * Y(:,170)) &
                + (RC(:,444) * Y(:,155)) + (RC(:,445) * Y(:,156)) + (RC(:,446) * Y(:,157)) &
                + (RC(:,441) * Y(:,172)) + (RC(:,442) * Y(:,173)) + (RC(:,443) * Y(:,154)) &
                + (RC(:,438) * Y(:,166)) + (RC(:,439) * Y(:,167)) + (RC(:,440) * Y(:,168)) &
                + (RC(:,435) * Y(:,163)) + (RC(:,436) * Y(:,164)) + (RC(:,437) * Y(:,165)) &
                + (RC(:,432) * Y(:,160)) + (RC(:,433) * Y(:,161)) + (RC(:,434) * Y(:,162)) &
                + (RC(:,429) * Y(:,149)) + (RC(:,430) * Y(:,150)) + (RC(:,431) * Y(:,159)) &
                + (RC(:,426) * Y(:,146)) + (RC(:,427) * Y(:,147)) + (RC(:,428) * Y(:,148)) &
                + (RC(:,423) * Y(:,144)) + (RC(:,424) * Y(:,144)) + (RC(:,425) * Y(:,145)) &
                + (RC(:,416) * Y(:,195)) + (RC(:,418) * Y(:,66)) + (RC(:,421) * Y(:,197)) &
                + (RC(:,411) * Y(:,142)) + (RC(:,412) * Y(:,143)) + (RC(:,413) * Y(:,63)) &
                + (RC(:,408) * Y(:,139)) + (RC(:,409) * Y(:,140)) + (RC(:,410) * Y(:,141)) &
                + (RC(:,405) * Y(:,136)) + (RC(:,406) * Y(:,137)) + (RC(:,407) * Y(:,138)) &
                + (RC(:,402) * Y(:,133)) + (RC(:,403) * Y(:,134)) + (RC(:,404) * Y(:,135)) &
                + (RC(:,399) * Y(:,130)) + (RC(:,400) * Y(:,131)) + (RC(:,401) * Y(:,132)) &
                + (RC(:,396) * Y(:,127)) + (RC(:,397) * Y(:,128)) + (RC(:,398) * Y(:,129)) &
                + (RC(:,393) * Y(:,124)) + (RC(:,394) * Y(:,125)) + (RC(:,395) * Y(:,126)) &
                + (RC(:,390) * Y(:,57)) + (RC(:,391) * Y(:,58)) + (RC(:,392) * Y(:,123)) &
                + (RC(:,385) * Y(:,193)) + (RC(:,386) * Y(:,120)) + (RC(:,389) * Y(:,52)) &
                + (RC(:,382) * Y(:,99)) + (RC(:,383) * Y(:,99)) + (RC(:,384) * Y(:,51)) &
                + (RC(:,379) * Y(:,96)) + (RC(:,380) * Y(:,97)) + (RC(:,381) * Y(:,97)) &
                + (RC(:,376) * Y(:,113)) + (RC(:,377) * Y(:,115)) + (RC(:,378) * Y(:,96)) &
                + (RC(:,370) * Y(:,190)) + (RC(:,371) * Y(:,191)) + (RC(:,372) * Y(:,109)) &
                + (RC(:,367) * Y(:,98)) + (RC(:,368) * Y(:,100)) + (RC(:,369) * Y(:,189)) &
                + (RC(:,360) * Y(:,46)) + (RC(:,364) * Y(:,102)) + (RC(:,366) * Y(:,60)) &
                + (RC(:,357) * Y(:,188)) + (RC(:,358) * Y(:,104)) + (RC(:,359) * Y(:,105)) &
                + (RC(:,354) * Y(:,187)) + (RC(:,355) * Y(:,88)) + (RC(:,356) * Y(:,111)) &
                + (RC(:,105) * Y(:,86)) + (RC(:,106) * Y(:,87)) + (RC(:,353) * Y(:,186)) &
                + (RC(:,102) * Y(:,83)) + (RC(:,103) * Y(:,84)) + (RC(:,104) * Y(:,85)) &
                + (RC(:,99) * Y(:,80)) + (RC(:,100) * Y(:,81)) + (RC(:,101) * Y(:,82)) &
                + (RC(:,96) * Y(:,79)) + (RC(:,97) * Y(:,40)) + (RC(:,98) * Y(:,41)) &
                + (RC(:,93) * Y(:,78)) + (RC(:,94) * Y(:,78)) + (RC(:,95) * Y(:,79)) &
                + (RC(:,90) * Y(:,76)) + (RC(:,91) * Y(:,77)) + (RC(:,92) * Y(:,77)) &
                + (RC(:,84) * Y(:,71)) + (RC(:,88) * Y(:,73)) + (RC(:,89) * Y(:,101)) &
                + (RC(:,81) * Y(:,67)) + (RC(:,82) * Y(:,39)) + (RC(:,83) * Y(:,42)) &
                + (RC(:,78) * Y(:,64)) + (RC(:,79) * Y(:,64)) + (RC(:,80) * Y(:,67)) &
                + (RC(:,75) * Y(:,59)) + (RC(:,76) * Y(:,61)) + (RC(:,77) * Y(:,61)) &
                + (RC(:,63) * Y(:,47)) + (RC(:,68) * Y(:,53)) + (RC(:,74) * Y(:,59)) &
                + (RC(:,48) * Y(:,32)) + (RC(:,49) * Y(:,34)) + (RC(:,59) * Y(:,43)) &
                + (RC(:,45) * Y(:,25)) + (RC(:,46) * Y(:,28)) + (RC(:,47) * Y(:,30)) &
                + (RC(:,42) * Y(:,21)) + (RC(:,43) * Y(:,23)) + (RC(:,44) * Y(:,25)) &
                + (RC(:,34) * Y(:,13)) + (RC(:,35) * Y(:,14)) + (RC(:,37) * Y(:,16)) &
                + (RC(:,27) * Y(:,4)) + (RC(:,28) * Y(:,5)) + (RC(:,32) * Y(:,15)) &
                + (RC(:,20) * Y(:,12)) + (RC(:,22) * Y(:,9)) + (RC(:,25) * Y(:,8)) &
                + (RC(:,17) * Y(:,6)) + (RC(:,18) * Y(:,10)) + (RC(:,19) * Y(:,11)) 
                Y(:,3) = P(:)/L(:)

                !          NO2              Y(:,4) 
                P(:) = (DJ(:,86) * Y(:,175)) + (DJ(:,96) * Y(:,185)) &
                + (DJ(:,80) * Y(:,170)) + (DJ(:,81) * Y(:,171)) &
                + (DJ(:,52) * Y(:,142)) + (DJ(:,79) * Y(:,169)) &
                + (DJ(:,50) * Y(:,137)) + (DJ(:,51) * Y(:,138)) &
                + (DJ(:,48) * Y(:,129)) + (DJ(:,49) * Y(:,136)) &
                + (DJ(:,46) * Y(:,127)) + (DJ(:,47) * Y(:,128)) &
                + (DJ(:,44) * Y(:,126)) + (DJ(:,45) * Y(:,127)) &
                + (DJ(:,42) * Y(:,124)) + (DJ(:,43) * Y(:,125)) &
                + (DJ(:,32) * Y(:,115)) + (DJ(:,41) * Y(:,123)) &
                + (DJ(:,8) * Y(:,14)) + (DJ(:,31) * Y(:,115)) &
                + (RC(:,484) * Y(:,3) * Y(:,203)) + (DJ(:,6) * Y(:,5)) &
                + (RC(:,481) * Y(:,3) * Y(:,201)) + (RC(:,483) * Y(:,203)) &
                + (RC(:,479) * Y(:,202)) + (RC(:,480) * Y(:,3) * Y(:,202)) &
                + (RC(:,475) * Y(:,3) * Y(:,200)) + (RC(:,477) * Y(:,201)) &
                + (RC(:,473) * Y(:,3) * Y(:,198)) + (RC(:,474) * Y(:,3) * Y(:,199)) &
                + (RC(:,470) * Y(:,199)) + (RC(:,472) * Y(:,200)) &
                + (RC(:,456) * Y(:,3) * Y(:,175)) + (RC(:,468) * Y(:,198)) &
                + (RC(:,449) * Y(:,3) * Y(:,170)) + (RC(:,450) * Y(:,3) * Y(:,171)) &
                + (RC(:,422) * Y(:,5) * Y(:,197)) + (RC(:,448) * Y(:,3) * Y(:,169)) &
                + (RC(:,417) * Y(:,5) * Y(:,195)) + (RC(:,421) * Y(:,3) * Y(:,197)) &
                + (RC(:,412) * Y(:,3) * Y(:,143)) + (RC(:,416) * Y(:,3) * Y(:,195)) &
                + (RC(:,410) * Y(:,3) * Y(:,141)) + (RC(:,411) * Y(:,3) * Y(:,142)) &
                + (RC(:,408) * Y(:,3) * Y(:,139)) + (RC(:,409) * Y(:,3) * Y(:,140)) &
                + (RC(:,406) * Y(:,3) * Y(:,137)) + (RC(:,407) * Y(:,3) * Y(:,138)) &
                + (RC(:,404) * Y(:,3) * Y(:,135)) + (RC(:,405) * Y(:,3) * Y(:,136)) &
                + (RC(:,402) * Y(:,3) * Y(:,133)) + (RC(:,403) * Y(:,3) * Y(:,134)) &
                + (RC(:,400) * Y(:,3) * Y(:,131)) + (RC(:,401) * Y(:,3) * Y(:,132)) &
                + (RC(:,398) * Y(:,3) * Y(:,129)) + (RC(:,399) * Y(:,3) * Y(:,130)) &
                + (RC(:,396) * Y(:,3) * Y(:,127)) + (RC(:,397) * Y(:,3) * Y(:,128)) &
                + (RC(:,394) * Y(:,3) * Y(:,125)) + (RC(:,395) * Y(:,3) * Y(:,126)) &
                + (RC(:,392) * Y(:,3) * Y(:,123)) + (RC(:,393) * Y(:,3) * Y(:,124)) &
                + (RC(:,352) * Y(:,55)) + (RC(:,377) * Y(:,3) * Y(:,115)) &
                + (RC(:,338) * Y(:,38)) + (RC(:,342) * Y(:,49)) &
                + (RC(:,336) * Y(:,36)) + (RC(:,337) * Y(:,37)) &
                + (RC(:,242) * Y(:,122) * Y(:,5)) &
                + (RC(:,243) * Y(:,55) * Y(:,5) * 2.00) &
                + (RC(:,240) * Y(:,54) * Y(:,5)) + (RC(:,241) * Y(:,56) * Y(:,5)) &
                + (RC(:,238) * Y(:,119) * Y(:,5)) + (RC(:,239) * Y(:,121) * Y(:,5)) &
                + (RC(:,236) * Y(:,117) * Y(:,5)) + (RC(:,237) * Y(:,118) * Y(:,5)) &
                + (RC(:,234) * Y(:,50) * Y(:,5)) + (RC(:,235) * Y(:,116) * Y(:,5)) &
                + (RC(:,232) * Y(:,48) * Y(:,5)) + (RC(:,233) * Y(:,49) * Y(:,5) * 2.00) &
                + (RC(:,230) * Y(:,45) * Y(:,5)) + (RC(:,231) * Y(:,114) * Y(:,5)) &
                + (RC(:,229) * Y(:,38) * Y(:,5) * 2.00) &
                + (RC(:,228) * Y(:,37) * Y(:,5) * 2.00) &
                + (RC(:,226) * Y(:,112) * Y(:,5)) &
                + (RC(:,227) * Y(:,36) * Y(:,5) * 2.00) &
                + (RC(:,224) * Y(:,112) * Y(:,5)) + (RC(:,225) * Y(:,112) * Y(:,5)) &
                + (RC(:,222) * Y(:,110) * Y(:,5)) + (RC(:,223) * Y(:,110) * Y(:,5)) &
                + (RC(:,220) * Y(:,44) * Y(:,5)) + (RC(:,221) * Y(:,44) * Y(:,5)) &
                + (RC(:,218) * Y(:,107) * Y(:,5)) + (RC(:,219) * Y(:,108) * Y(:,5)) &
                + (RC(:,216) * Y(:,74) * Y(:,5)) + (RC(:,217) * Y(:,75) * Y(:,5)) &
                + (RC(:,214) * Y(:,72) * Y(:,5)) + (RC(:,215) * Y(:,106) * Y(:,5)) &
                + (RC(:,212) * Y(:,92) * Y(:,5)) + (RC(:,213) * Y(:,70) * Y(:,5)) &
                + (RC(:,210) * Y(:,103) * Y(:,5)) + (RC(:,211) * Y(:,90) * Y(:,5)) &
                + (RC(:,208) * Y(:,35) * Y(:,5)) + (RC(:,209) * Y(:,95) * Y(:,5)) &
                + (RC(:,206) * Y(:,31) * Y(:,5)) + (RC(:,207) * Y(:,33) * Y(:,5)) &
                + (RC(:,204) * Y(:,69) * Y(:,5)) + (RC(:,205) * Y(:,31) * Y(:,5)) &
                + (RC(:,202) * Y(:,65) * Y(:,5)) + (RC(:,203) * Y(:,68) * Y(:,5)) &
                + (RC(:,200) * Y(:,62) * Y(:,5)) + (RC(:,201) * Y(:,65) * Y(:,5)) &
                + (RC(:,198) * Y(:,93) * Y(:,5)) + (RC(:,199) * Y(:,94) * Y(:,5)) &
                + (RC(:,196) * Y(:,89) * Y(:,5)) + (RC(:,197) * Y(:,91) * Y(:,5)) &
                + (RC(:,194) * Y(:,29) * Y(:,5)) + (RC(:,195) * Y(:,29) * Y(:,5)) &
                + (RC(:,192) * Y(:,27) * Y(:,5)) + (RC(:,193) * Y(:,26) * Y(:,5)) &
                + (RC(:,190) * Y(:,22) * Y(:,5)) + (RC(:,191) * Y(:,24) * Y(:,5)) &
                + (RC(:,163) * Y(:,122) * Y(:,8)) + (RC(:,165) * Y(:,217)) &
                + (RC(:,161) * Y(:,56) * Y(:,8)) + (RC(:,162) * Y(:,56) * Y(:,8)) &
                + (RC(:,160) * Y(:,55) * Y(:,8) * 2.00) &
                + (RC(:,158) * Y(:,54) * Y(:,8)) + (RC(:,159) * Y(:,54) * Y(:,8)) &
                + (RC(:,156) * Y(:,119) * Y(:,8)) + (RC(:,157) * Y(:,121) * Y(:,8)) &
                + (RC(:,154) * Y(:,117) * Y(:,8)) + (RC(:,155) * Y(:,118) * Y(:,8)) &
                + (RC(:,152) * Y(:,50) * Y(:,8)) + (RC(:,153) * Y(:,116) * Y(:,8)) &
                + (RC(:,151) * Y(:,49) * Y(:,8) * 2.00) &
                + (RC(:,149) * Y(:,48) * Y(:,8)) + (RC(:,150) * Y(:,48) * Y(:,8)) &
                + (RC(:,147) * Y(:,45) * Y(:,8)) + (RC(:,148) * Y(:,114) * Y(:,8)) &
                + (RC(:,146) * Y(:,38) * Y(:,8) * 2.00) &
                + (RC(:,145) * Y(:,37) * Y(:,8) * 2.00) &
                + (RC(:,143) * Y(:,112) * Y(:,8)) &
                + (RC(:,144) * Y(:,36) * Y(:,8) * 2.00) &
                + (RC(:,141) * Y(:,112) * Y(:,8)) + (RC(:,142) * Y(:,112) * Y(:,8)) &
                + (RC(:,139) * Y(:,110) * Y(:,8)) + (RC(:,140) * Y(:,110) * Y(:,8)) &
                + (RC(:,137) * Y(:,44) * Y(:,8)) + (RC(:,138) * Y(:,44) * Y(:,8)) &
                + (RC(:,135) * Y(:,107) * Y(:,8)) + (RC(:,136) * Y(:,108) * Y(:,8)) &
                + (RC(:,133) * Y(:,74) * Y(:,8)) + (RC(:,134) * Y(:,75) * Y(:,8)) &
                + (RC(:,131) * Y(:,72) * Y(:,8)) + (RC(:,132) * Y(:,106) * Y(:,8)) &
                + (RC(:,129) * Y(:,92) * Y(:,8)) + (RC(:,130) * Y(:,70) * Y(:,8)) &
                + (RC(:,127) * Y(:,103) * Y(:,8)) + (RC(:,128) * Y(:,90) * Y(:,8)) &
                + (RC(:,125) * Y(:,35) * Y(:,8)) + (RC(:,126) * Y(:,95) * Y(:,8)) &
                + (RC(:,123) * Y(:,31) * Y(:,8)) + (RC(:,124) * Y(:,33) * Y(:,8)) &
                + (RC(:,121) * Y(:,69) * Y(:,8)) + (RC(:,122) * Y(:,31) * Y(:,8)) &
                + (RC(:,119) * Y(:,65) * Y(:,8)) + (RC(:,120) * Y(:,68) * Y(:,8)) &
                + (RC(:,117) * Y(:,62) * Y(:,8)) + (RC(:,118) * Y(:,65) * Y(:,8)) &
                + (RC(:,115) * Y(:,93) * Y(:,8)) + (RC(:,116) * Y(:,94) * Y(:,8)) &
                + (RC(:,113) * Y(:,89) * Y(:,8)) + (RC(:,114) * Y(:,91) * Y(:,8)) &
                + (RC(:,111) * Y(:,29) * Y(:,8)) + (RC(:,112) * Y(:,29) * Y(:,8)) &
                + (RC(:,109) * Y(:,27) * Y(:,8)) + (RC(:,110) * Y(:,26) * Y(:,8)) &
                + (RC(:,107) * Y(:,22) * Y(:,8)) + (RC(:,108) * Y(:,24) * Y(:,8)) &
                + (RC(:,33) * Y(:,9) * Y(:,5)) + (RC(:,34) * Y(:,3) * Y(:,13)) &
                + (RC(:,31) * Y(:,15)) + (RC(:,32) * Y(:,3) * Y(:,15)) &
                + (RC(:,28) * Y(:,3) * Y(:,5)) + (RC(:,29) * Y(:,9) * Y(:,8)) &
                + (RC(:,13) * Y(:,4) * Y(:,5)) + (RC(:,15) * Y(:,7)) &
                + (RC(:,12) * Y(:,8) * Y(:,5) * 2.00) &
                + (RC(:,11) * Y(:,8) * Y(:,8) * 2.00) &
                + (RC(:,4) * Y(:,2) * Y(:,8)) + (RC(:,9) * Y(:,8) * Y(:,6)) 
                !     
                !          
                L(:) = 0.0 &
                + (RC(:,478) * Y(:,112)) + (RC(:,482) * Y(:,50)) + (DJ(:,4)) &
                + (RC(:,469) * Y(:,72)) + (RC(:,471) * Y(:,106)) + (RC(:,476) * Y(:,110)) &
                + (RC(:,415) * Y(:,194)) + (RC(:,420) * Y(:,196)) + (RC(:,467) * Y(:,70)) &
                + (RC(:,27) * Y(:,3)) + (RC(:,30) * Y(:,9)) + (RC(:,164) * Y(:,22)) &
                + (RC(:,13) * Y(:,5)) + (RC(:,14) * Y(:,5)) + (RC(:,26)) &
                + (RC(:,5) * Y(:,2)) + (RC(:,6) * Y(:,2)) + (RC(:,10) * Y(:,6)) 
                Y(:,4) = (YP(:,4) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NO3              Y(:,5) 
                P(:) = (RC(:,15) * Y(:,7)) + (RC(:,35) * Y(:,3) * Y(:,14)) &
                + (RC(:,6) * Y(:,2) * Y(:,4)) + (RC(:,10) * Y(:,4) * Y(:,6)) 
                L(:) = 0.0 &
                + (DJ(:,6)) &
                + (RC(:,419) * Y(:,66)) + (RC(:,422) * Y(:,197)) + (DJ(:,5)) &
                + (RC(:,388) * Y(:,120)) + (RC(:,414) * Y(:,63)) + (RC(:,417) * Y(:,195)) &
                + (RC(:,365) * Y(:,102)) + (RC(:,373) * Y(:,109)) + (RC(:,387) * Y(:,51)) &
                + (RC(:,242) * Y(:,122)) + (RC(:,243) * Y(:,55)) + (RC(:,361) * Y(:,46)) &
                + (RC(:,239) * Y(:,121)) + (RC(:,240) * Y(:,54)) + (RC(:,241) * Y(:,56)) &
                + (RC(:,236) * Y(:,117)) + (RC(:,237) * Y(:,118)) + (RC(:,238) * Y(:,119)) &
                + (RC(:,233) * Y(:,49)) + (RC(:,234) * Y(:,50)) + (RC(:,235) * Y(:,116)) &
                + (RC(:,230) * Y(:,45)) + (RC(:,231) * Y(:,114)) + (RC(:,232) * Y(:,48)) &
                + (RC(:,227) * Y(:,36)) + (RC(:,228) * Y(:,37)) + (RC(:,229) * Y(:,38)) &
                + (RC(:,224) * Y(:,112)) + (RC(:,225) * Y(:,112)) + (RC(:,226) * Y(:,112)) &
                + (RC(:,221) * Y(:,44)) + (RC(:,222) * Y(:,110)) + (RC(:,223) * Y(:,110)) &
                + (RC(:,218) * Y(:,107)) + (RC(:,219) * Y(:,108)) + (RC(:,220) * Y(:,44)) &
                + (RC(:,215) * Y(:,106)) + (RC(:,216) * Y(:,74)) + (RC(:,217) * Y(:,75)) &
                + (RC(:,212) * Y(:,92)) + (RC(:,213) * Y(:,70)) + (RC(:,214) * Y(:,72)) &
                + (RC(:,209) * Y(:,95)) + (RC(:,210) * Y(:,103)) + (RC(:,211) * Y(:,90)) &
                + (RC(:,206) * Y(:,31)) + (RC(:,207) * Y(:,33)) + (RC(:,208) * Y(:,35)) &
                + (RC(:,203) * Y(:,68)) + (RC(:,204) * Y(:,69)) + (RC(:,205) * Y(:,31)) &
                + (RC(:,200) * Y(:,62)) + (RC(:,201) * Y(:,65)) + (RC(:,202) * Y(:,65)) &
                + (RC(:,197) * Y(:,91)) + (RC(:,198) * Y(:,93)) + (RC(:,199) * Y(:,94)) &
                + (RC(:,194) * Y(:,29)) + (RC(:,195) * Y(:,29)) + (RC(:,196) * Y(:,89)) &
                + (RC(:,191) * Y(:,24)) + (RC(:,192) * Y(:,27)) + (RC(:,193) * Y(:,26)) &
                + (RC(:,86) * Y(:,42)) + (RC(:,87) * Y(:,71)) + (RC(:,190) * Y(:,22)) &
                + (RC(:,64) * Y(:,47)) + (RC(:,69) * Y(:,53)) + (RC(:,85) * Y(:,39)) &
                + (RC(:,51) * Y(:,32)) + (RC(:,52) * Y(:,34)) + (RC(:,60) * Y(:,43)) &
                + (RC(:,28) * Y(:,3)) + (RC(:,33) * Y(:,9)) + (RC(:,50) * Y(:,30)) &
                + (RC(:,12) * Y(:,8)) + (RC(:,13) * Y(:,4)) + (RC(:,14) * Y(:,4)) 
                Y(:,5) = (YP(:,5) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          O3               Y(:,6) 
                P(:) = (RC(:,1) * Y(:,2)) + (RC(:,2) * Y(:,2)) 
                !     

                L(:) = 0.0 &
                + (DJ(:,1)) + (DJ(:,2)) &
                + (RC(:,363) * Y(:,46)) + (RC(:,374) * Y(:,109)) + (RC(:,375) * Y(:,109)) &
                + (RC(:,72) * Y(:,53)) + (RC(:,73) * Y(:,53)) + (RC(:,362) * Y(:,46)) &
                + (RC(:,67) * Y(:,47)) + (RC(:,70) * Y(:,53)) + (RC(:,71) * Y(:,53)) &
                + (RC(:,62) * Y(:,43)) + (RC(:,65) * Y(:,47)) + (RC(:,66) * Y(:,47)) &
                + (RC(:,57) * Y(:,34)) + (RC(:,58) * Y(:,34)) + (RC(:,61) * Y(:,43)) &
                + (RC(:,54) * Y(:,30)) + (RC(:,55) * Y(:,32)) + (RC(:,56) * Y(:,32)) &
                + (RC(:,17) * Y(:,3)) + (RC(:,21) * Y(:,9)) + (RC(:,53) * Y(:,30)) &
                + (RC(:,3) * Y(:,2)) + (RC(:,9) * Y(:,8)) + (RC(:,10) * Y(:,4)) 
                Y(:,6) = (YP(:,6) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          N2O5             Y(:,7) 
                P(:) = (RC(:,14) * Y(:,4) * Y(:,5)) 
                !   

                L(:) = 0.0 &
                + (RC(:,15)) + (RC(:,40)) 
                Y(:,7) = (YP(:,7) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NO               Y(:,8) 
                P(:) = (DJ(:,7) * Y(:,13)) &
                + (DJ(:,4) * Y(:,4)) + (DJ(:,5) * Y(:,5)) &
                + (RC(:,5) * Y(:,2) * Y(:,4)) + (RC(:,13) * Y(:,4) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,189) * Y(:,122)) &
                + (RC(:,186) * Y(:,116)) + (RC(:,187) * Y(:,54)) + (RC(:,188) * Y(:,56)) &
                + (RC(:,183) * Y(:,68)) + (RC(:,184) * Y(:,69)) + (RC(:,185) * Y(:,48)) &
                + (RC(:,180) * Y(:,44)) + (RC(:,181) * Y(:,62)) + (RC(:,182) * Y(:,65)) &
                + (RC(:,177) * Y(:,103)) + (RC(:,178) * Y(:,90)) + (RC(:,179) * Y(:,92)) &
                + (RC(:,174) * Y(:,33)) + (RC(:,175) * Y(:,35)) + (RC(:,176) * Y(:,95)) &
                + (RC(:,171) * Y(:,89)) + (RC(:,172) * Y(:,91)) + (RC(:,173) * Y(:,31)) &
                + (RC(:,168) * Y(:,27)) + (RC(:,169) * Y(:,26)) + (RC(:,170) * Y(:,29)) &
                + (RC(:,163) * Y(:,122)) + (RC(:,166) * Y(:,22)) + (RC(:,167) * Y(:,24)) &
                + (RC(:,160) * Y(:,55)) + (RC(:,161) * Y(:,56)) + (RC(:,162) * Y(:,56)) &
                + (RC(:,157) * Y(:,121)) + (RC(:,158) * Y(:,54)) + (RC(:,159) * Y(:,54)) &
                + (RC(:,154) * Y(:,117)) + (RC(:,155) * Y(:,118)) + (RC(:,156) * Y(:,119)) &
                + (RC(:,151) * Y(:,49)) + (RC(:,152) * Y(:,50)) + (RC(:,153) * Y(:,116)) &
                + (RC(:,148) * Y(:,114)) + (RC(:,149) * Y(:,48)) + (RC(:,150) * Y(:,48)) &
                + (RC(:,145) * Y(:,37)) + (RC(:,146) * Y(:,38)) + (RC(:,147) * Y(:,45)) &
                + (RC(:,142) * Y(:,112)) + (RC(:,143) * Y(:,112)) + (RC(:,144) * Y(:,36)) &
                + (RC(:,139) * Y(:,110)) + (RC(:,140) * Y(:,110)) + (RC(:,141) * Y(:,112)) &
                + (RC(:,136) * Y(:,108)) + (RC(:,137) * Y(:,44)) + (RC(:,138) * Y(:,44)) &
                + (RC(:,133) * Y(:,74)) + (RC(:,134) * Y(:,75)) + (RC(:,135) * Y(:,107)) &
                + (RC(:,130) * Y(:,70)) + (RC(:,131) * Y(:,72)) + (RC(:,132) * Y(:,106)) &
                + (RC(:,127) * Y(:,103)) + (RC(:,128) * Y(:,90)) + (RC(:,129) * Y(:,92)) &
                + (RC(:,124) * Y(:,33)) + (RC(:,125) * Y(:,35)) + (RC(:,126) * Y(:,95)) &
                + (RC(:,121) * Y(:,69)) + (RC(:,122) * Y(:,31)) + (RC(:,123) * Y(:,31)) &
                + (RC(:,118) * Y(:,65)) + (RC(:,119) * Y(:,65)) + (RC(:,120) * Y(:,68)) &
                + (RC(:,115) * Y(:,93)) + (RC(:,116) * Y(:,94)) + (RC(:,117) * Y(:,62)) &
                + (RC(:,112) * Y(:,29)) + (RC(:,113) * Y(:,89)) + (RC(:,114) * Y(:,91)) &
                + (RC(:,109) * Y(:,27)) + (RC(:,110) * Y(:,26)) + (RC(:,111) * Y(:,29)) &
                + (RC(:,29) * Y(:,9)) + (RC(:,107) * Y(:,22)) + (RC(:,108) * Y(:,24)) &
                + (RC(:,11) * Y(:,8)) + (RC(:,12) * Y(:,5)) + (RC(:,25) * Y(:,3)) &
                + (RC(:,4) * Y(:,2)) + (RC(:,9) * Y(:,6)) + (RC(:,11) * Y(:,8)) 
                Y(:,8) = (YP(:,8) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HO2              Y(:,9) 
                P(:) = (DJ(:,94) * Y(:,183)) &
                + (DJ(:,91) * Y(:,180)) + (DJ(:,93) * Y(:,182)) &
                + (DJ(:,84) * Y(:,153)) + (DJ(:,85) * Y(:,174)) &
                + (DJ(:,82) * Y(:,151)) + (DJ(:,83) * Y(:,152)) &
                + (DJ(:,77) * Y(:,157)) + (DJ(:,78) * Y(:,158)) &
                + (DJ(:,75) * Y(:,155)) + (DJ(:,76) * Y(:,156)) &
                + (DJ(:,73) * Y(:,173)) + (DJ(:,74) * Y(:,154)) &
                + (DJ(:,70) * Y(:,167)) + (DJ(:,72) * Y(:,172)) &
                + (DJ(:,68) * Y(:,166)) + (DJ(:,69) * Y(:,166)) &
                + (DJ(:,58) * Y(:,148)) + (DJ(:,63) * Y(:,161)) &
                + (DJ(:,55) * Y(:,146)) + (DJ(:,56) * Y(:,147)) &
                + (DJ(:,53) * Y(:,144)) + (DJ(:,54) * Y(:,145)) &
                + (DJ(:,51) * Y(:,138)) + (DJ(:,52) * Y(:,142)) &
                + (DJ(:,49) * Y(:,136)) + (DJ(:,50) * Y(:,137)) &
                + (DJ(:,44) * Y(:,126)) + (DJ(:,46) * Y(:,127)) &
                + (DJ(:,42) * Y(:,124)) + (DJ(:,43) * Y(:,125)) &
                + (DJ(:,39) * Y(:,51)) + (DJ(:,41) * Y(:,123)) &
                + (DJ(:,37) * Y(:,99)) + (DJ(:,38) * Y(:,99)) &
                + (DJ(:,35) * Y(:,97)) + (DJ(:,36) * Y(:,97)) &
                + (DJ(:,33) * Y(:,96)) + (DJ(:,34) * Y(:,96) * 2.00) &
                + (DJ(:,30) * Y(:,113) * 2.00) &
                + (DJ(:,25) * Y(:,98)) + (DJ(:,29) * Y(:,109)) &
                + (DJ(:,23) * Y(:,46)) + (DJ(:,24) * Y(:,60) * 2.00) &
                + (DJ(:,22) * Y(:,102) * 2.00) &
                + (DJ(:,20) * Y(:,104)) + (DJ(:,21) * Y(:,105)) &
                + (DJ(:,18) * Y(:,111)) + (DJ(:,19) * Y(:,188)) &
                + (DJ(:,11) * Y(:,42)) + (DJ(:,12) * Y(:,71)) &
                + (RC(:,379) * Y(:,3) * Y(:,96)) + (DJ(:,9) * Y(:,39) * 2.00) &
                + (RC(:,357) * Y(:,3) * Y(:,188)) + (RC(:,366) * Y(:,3) * Y(:,60)) &
                + (RC(:,350) * Y(:,56)) + (RC(:,356) * Y(:,3) * Y(:,111)) &
                + (RC(:,347) * Y(:,119)) + (RC(:,349) * Y(:,54)) &
                + (RC(:,340) * Y(:,114)) + (RC(:,341) * Y(:,48)) &
                + (RC(:,335) * Y(:,112)) + (RC(:,339) * Y(:,45)) &
                + (RC(:,332) * Y(:,110)) + (RC(:,334) * Y(:,112)) &
                + (RC(:,329) * Y(:,44)) + (RC(:,330) * Y(:,44)) &
                + (RC(:,321) * Y(:,92)) + (RC(:,324) * Y(:,106)) &
                + (RC(:,319) * Y(:,103)) + (RC(:,320) * Y(:,90)) &
                + (RC(:,317) * Y(:,35)) + (RC(:,318) * Y(:,95)) &
                + (RC(:,315) * Y(:,31)) + (RC(:,316) * Y(:,33)) &
                + (RC(:,311) * Y(:,69)) + (RC(:,314) * Y(:,31)) &
                + (RC(:,309) * Y(:,65)) + (RC(:,310) * Y(:,68)) &
                + (RC(:,307) * Y(:,62)) + (RC(:,308) * Y(:,65)) &
                + (RC(:,300) * Y(:,26)) + (RC(:,304) * Y(:,29)) &
                + (RC(:,294) * Y(:,24)) + (RC(:,297) * Y(:,27)) &
                + (RC(:,241) * Y(:,56) * Y(:,5)) + (RC(:,291) * Y(:,22)) &
                + (RC(:,238) * Y(:,119) * Y(:,5)) + (RC(:,240) * Y(:,54) * Y(:,5)) &
                + (RC(:,231) * Y(:,114) * Y(:,5)) + (RC(:,232) * Y(:,48) * Y(:,5)) &
                + (RC(:,226) * Y(:,112) * Y(:,5)) + (RC(:,230) * Y(:,45) * Y(:,5)) &
                + (RC(:,223) * Y(:,110) * Y(:,5)) + (RC(:,225) * Y(:,112) * Y(:,5)) &
                + (RC(:,220) * Y(:,44) * Y(:,5)) + (RC(:,221) * Y(:,44) * Y(:,5)) &
                + (RC(:,212) * Y(:,92) * Y(:,5)) + (RC(:,215) * Y(:,106) * Y(:,5)) &
                + (RC(:,210) * Y(:,103) * Y(:,5)) + (RC(:,211) * Y(:,90) * Y(:,5)) &
                + (RC(:,208) * Y(:,35) * Y(:,5)) + (RC(:,209) * Y(:,95) * Y(:,5)) &
                + (RC(:,206) * Y(:,31) * Y(:,5)) + (RC(:,207) * Y(:,33) * Y(:,5)) &
                + (RC(:,204) * Y(:,69) * Y(:,5)) + (RC(:,205) * Y(:,31) * Y(:,5)) &
                + (RC(:,202) * Y(:,65) * Y(:,5)) + (RC(:,203) * Y(:,68) * Y(:,5)) &
                + (RC(:,200) * Y(:,62) * Y(:,5)) + (RC(:,201) * Y(:,65) * Y(:,5)) &
                + (RC(:,193) * Y(:,26) * Y(:,5)) + (RC(:,195) * Y(:,29) * Y(:,5)) &
                + (RC(:,191) * Y(:,24) * Y(:,5)) + (RC(:,192) * Y(:,27) * Y(:,5)) &
                + (RC(:,161) * Y(:,56) * Y(:,8)) + (RC(:,190) * Y(:,22) * Y(:,5)) &
                + (RC(:,156) * Y(:,119) * Y(:,8)) + (RC(:,158) * Y(:,54) * Y(:,8)) &
                + (RC(:,148) * Y(:,114) * Y(:,8)) + (RC(:,149) * Y(:,48) * Y(:,8)) &
                + (RC(:,143) * Y(:,112) * Y(:,8)) + (RC(:,147) * Y(:,45) * Y(:,8)) &
                + (RC(:,140) * Y(:,110) * Y(:,8)) + (RC(:,142) * Y(:,112) * Y(:,8)) &
                + (RC(:,137) * Y(:,44) * Y(:,8)) + (RC(:,138) * Y(:,44) * Y(:,8)) &
                + (RC(:,129) * Y(:,92) * Y(:,8)) + (RC(:,132) * Y(:,106) * Y(:,8)) &
                + (RC(:,127) * Y(:,103) * Y(:,8)) + (RC(:,128) * Y(:,90) * Y(:,8)) &
                + (RC(:,125) * Y(:,35) * Y(:,8)) + (RC(:,126) * Y(:,95) * Y(:,8)) &
                + (RC(:,123) * Y(:,31) * Y(:,8)) + (RC(:,124) * Y(:,33) * Y(:,8)) &
                + (RC(:,121) * Y(:,69) * Y(:,8)) + (RC(:,122) * Y(:,31) * Y(:,8)) &
                + (RC(:,119) * Y(:,65) * Y(:,8)) + (RC(:,120) * Y(:,68) * Y(:,8)) &
                + (RC(:,117) * Y(:,62) * Y(:,8)) + (RC(:,118) * Y(:,65) * Y(:,8)) &
                + (RC(:,110) * Y(:,26) * Y(:,8)) + (RC(:,112) * Y(:,29) * Y(:,8)) &
                + (RC(:,108) * Y(:,24) * Y(:,8)) + (RC(:,109) * Y(:,27) * Y(:,8)) &
                + (RC(:,97) * Y(:,40) * Y(:,3)) + (RC(:,107) * Y(:,22) * Y(:,8)) &
                + (RC(:,93) * Y(:,78) * Y(:,3)) + (RC(:,95) * Y(:,3) * Y(:,79)) &
                + (RC(:,90) * Y(:,3) * Y(:,76)) + (RC(:,91) * Y(:,3) * Y(:,77)) &
                + (RC(:,82) * Y(:,3) * Y(:,39)) + (RC(:,85) * Y(:,5) * Y(:,39)) &
                + (RC(:,77) * Y(:,61) * Y(:,3)) + (RC(:,79) * Y(:,64) * Y(:,3)) &
                + (RC(:,61) * Y(:,6) * Y(:,43)) + (RC(:,74) * Y(:,59) * Y(:,3)) &
                + (RC(:,38) * Y(:,18)) + (RC(:,53) * Y(:,6) * Y(:,30)) &
                + (RC(:,28) * Y(:,3) * Y(:,5)) + (RC(:,31) * Y(:,15)) &
                + (RC(:,19) * Y(:,3) * Y(:,11)) + (RC(:,20) * Y(:,3) * Y(:,12)) &
                + (RC(:,17) * Y(:,3) * Y(:,6)) + (RC(:,18) * Y(:,3) * Y(:,10)) 
                L(:) = 0.0 &
                + (RC(:,289) * Y(:,122)) + (RC(:,290) * Y(:,55)) &
                + (RC(:,286) * Y(:,121)) + (RC(:,287) * Y(:,54)) + (RC(:,288) * Y(:,56)) &
                + (RC(:,283) * Y(:,117)) + (RC(:,284) * Y(:,118)) + (RC(:,285) * Y(:,119)) &
                + (RC(:,280) * Y(:,49)) + (RC(:,281) * Y(:,50)) + (RC(:,282) * Y(:,116)) &
                + (RC(:,277) * Y(:,45)) + (RC(:,278) * Y(:,114)) + (RC(:,279) * Y(:,48)) &
                + (RC(:,274) * Y(:,36)) + (RC(:,275) * Y(:,37)) + (RC(:,276) * Y(:,38)) &
                + (RC(:,271) * Y(:,44)) + (RC(:,272) * Y(:,110)) + (RC(:,273) * Y(:,112)) &
                + (RC(:,268) * Y(:,75)) + (RC(:,269) * Y(:,107)) + (RC(:,270) * Y(:,108)) &
                + (RC(:,265) * Y(:,72)) + (RC(:,266) * Y(:,106)) + (RC(:,267) * Y(:,74)) &
                + (RC(:,262) * Y(:,90)) + (RC(:,263) * Y(:,92)) + (RC(:,264) * Y(:,70)) &
                + (RC(:,259) * Y(:,35)) + (RC(:,260) * Y(:,95)) + (RC(:,261) * Y(:,103)) &
                + (RC(:,256) * Y(:,69)) + (RC(:,257) * Y(:,31)) + (RC(:,258) * Y(:,33)) &
                + (RC(:,253) * Y(:,62)) + (RC(:,254) * Y(:,65)) + (RC(:,255) * Y(:,68)) &
                + (RC(:,250) * Y(:,91)) + (RC(:,251) * Y(:,93)) + (RC(:,252) * Y(:,94)) &
                + (RC(:,247) * Y(:,26)) + (RC(:,248) * Y(:,29)) + (RC(:,249) * Y(:,89)) &
                + (RC(:,244) * Y(:,22)) + (RC(:,245) * Y(:,24)) + (RC(:,246) * Y(:,27)) &
                + (RC(:,29) * Y(:,8)) + (RC(:,30) * Y(:,4)) + (RC(:,33) * Y(:,5)) &
                + (RC(:,23) * Y(:,9)) + (RC(:,24) * Y(:,9)) + (RC(:,24) * Y(:,9)) &
                + (RC(:,21) * Y(:,6)) + (RC(:,22) * Y(:,3)) + (RC(:,23) * Y(:,9)) 
                Y(:,9) = (YP(:,9) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          H2               Y(:,10) 
                P(:) = (DJ(:,10) * Y(:,39)) 
                L(:) = 0.0 &
                + (RC(:,18) * Y(:,3)) 
                Y(:,10) = (YP(:,10) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CO               Y(:,11) 
                P(:) = (DJ(:,92) * Y(:,181)) &
                + (DJ(:,40) * Y(:,120)) + (DJ(:,73) * Y(:,173)) &
                + (DJ(:,30) * Y(:,113) * 2.00) &
                + (DJ(:,25) * Y(:,98)) + (DJ(:,29) * Y(:,109)) &
                + (DJ(:,24) * Y(:,60) * 2.00) &
                + (DJ(:,12) * Y(:,71)) + (DJ(:,22) * Y(:,102)) &
                + (DJ(:,10) * Y(:,39)) + (DJ(:,11) * Y(:,42)) &
                + (RC(:,480) * Y(:,3) * Y(:,202)) + (DJ(:,9) * Y(:,39)) &
                + (RC(:,474) * Y(:,3) * Y(:,199)) + (RC(:,475) * Y(:,3) * Y(:,200)) &
                + (RC(:,442) * Y(:,3) * Y(:,173)) + (RC(:,473) * Y(:,3) * Y(:,198)) &
                + (RC(:,367) * Y(:,3) * Y(:,98)) + (RC(:,374) * Y(:,6) * Y(:,109)) &
                + (RC(:,362) * Y(:,6) * Y(:,46)) + (RC(:,366) * Y(:,3) * Y(:,60) * 2.00) &
                + (RC(:,340) * Y(:,114)) + (RC(:,348) * Y(:,121)) &
                + (RC(:,231) * Y(:,114) * Y(:,5)) + (RC(:,239) * Y(:,121) * Y(:,5)) &
                + (RC(:,157) * Y(:,121) * Y(:,8)) + (RC(:,223) * Y(:,110) * Y(:,5)) &
                + (RC(:,140) * Y(:,110) * Y(:,8)) + (RC(:,148) * Y(:,114) * Y(:,8)) &
                + (RC(:,82) * Y(:,3) * Y(:,39)) + (RC(:,85) * Y(:,5) * Y(:,39)) &
                + (RC(:,73) * Y(:,53) * Y(:,6)) + (RC(:,74) * Y(:,59) * Y(:,3)) &
                + (RC(:,57) * Y(:,6) * Y(:,34)) + (RC(:,61) * Y(:,6) * Y(:,43)) &
                + (RC(:,53) * Y(:,6) * Y(:,30)) + (RC(:,55) * Y(:,6) * Y(:,32)) 
                L(:) = 0.0 &
                + (RC(:,19) * Y(:,3)) 
                Y(:,11) = (YP(:,11) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          H2O2             Y(:,12) 
                P(:) = (RC(:,363) * Y(:,6) * Y(:,46)) + (RC(:,375) * Y(:,6) * Y(:,109)) &
                + (RC(:,66) * Y(:,47) * Y(:,6)) + (RC(:,71) * Y(:,53) * Y(:,6)) &
                + (RC(:,23) * Y(:,9) * Y(:,9)) + (RC(:,24) * Y(:,9) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,20) * Y(:,3)) + (DJ(:,3)) 
                Y(:,12) = (YP(:,12) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HONO             Y(:,13) 
                P(:) = (RC(:,25) * Y(:,3) * Y(:,8)) + (RC(:,26) * Y(:,4)) 
                L(:) = 0.0 &
                + (RC(:,34) * Y(:,3)) + (DJ(:,7)) 
                Y(:,13) = (YP(:,13) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HNO3             Y(:,14) 
                P(:) = (RC(:,422) * Y(:,5) * Y(:,197)) &
                + (RC(:,417) * Y(:,5) * Y(:,195)) + (RC(:,419) * Y(:,5) * Y(:,66)) &
                + (RC(:,388) * Y(:,5) * Y(:,120)) + (RC(:,414) * Y(:,5) * Y(:,63)) &
                + (RC(:,373) * Y(:,5) * Y(:,109)) + (RC(:,387) * Y(:,5) * Y(:,51)) &
                + (RC(:,361) * Y(:,5) * Y(:,46)) + (RC(:,365) * Y(:,5) * Y(:,102)) &
                + (RC(:,86) * Y(:,5) * Y(:,42)) + (RC(:,87) * Y(:,5) * Y(:,71)) &
                + (RC(:,27) * Y(:,3) * Y(:,4)) + (RC(:,85) * Y(:,5) * Y(:,39)) 
                L(:) = 0.0 &
                + (RC(:,35) * Y(:,3)) + (RC(:,39)) + (DJ(:,8)) 
                Y(:,14) = (YP(:,14) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HO2NO2           Y(:,15) 
                P(:) = (RC(:,30) * Y(:,9) * Y(:,4)) 
                L(:) = 0.0 &
                + (RC(:,31)) + (RC(:,32) * Y(:,3)) 
                Y(:,15) = (YP(:,15) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          SO2              Y(:,16) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,36) * Y(:,2)) + (RC(:,37) * Y(:,3)) 
                Y(:,16) = (YP(:,16) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          SO3              Y(:,17) 
                P(:) = (RC(:,36) * Y(:,2) * Y(:,16)) + (RC(:,38) * Y(:,18)) 
                L(:) = 0.0 &
                + (RC(:,41)) 
                Y(:,17) = (YP(:,17) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HSO3             Y(:,18) 
                P(:) = (RC(:,37) * Y(:,3) * Y(:,16)) 
                L(:) = 0.0 &
                + (RC(:,38)) 
                Y(:,18) = (YP(:,18) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NA               Y(:,19) 
                P(:) = (RC(:,40) * Y(:,7)) &
                + (RC(:,39) * Y(:,14)) + (RC(:,40) * Y(:,7)) 
                L(:) = 0.0 
                Y(:,19) = (YP(:,19) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          SA               Y(:,20) 
                P(:) = (RC(:,41) * Y(:,17)) 
                L(:) = 0.0 
                Y(:,20) = (YP(:,20) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH4              Y(:,21) 
                P(:) = 0.0
                L(:) = 0.0 &
                + (RC(:,42) * Y(:,3)) 
                Y(:,21) = (YP(:,21) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3O2            Y(:,22) 
                P(:) = (DJ(:,61) * Y(:,159)) &
                + (DJ(:,13) * Y(:,73)) + (DJ(:,36) * Y(:,97)) &
                + (RC(:,423) * Y(:,3) * Y(:,144)) + (DJ(:,11) * Y(:,42)) &
                + (RC(:,322) * Y(:,70)) + (RC(:,381) * Y(:,3) * Y(:,97)) &
                + (RC(:,165) * Y(:,217)) + (RC(:,213) * Y(:,70) * Y(:,5)) &
                + (RC(:,101) * Y(:,3) * Y(:,82)) + (RC(:,130) * Y(:,70) * Y(:,8)) &
                + (RC(:,99) * Y(:,3) * Y(:,80)) + (RC(:,100) * Y(:,3) * Y(:,81)) &
                + (RC(:,57) * Y(:,6) * Y(:,34)) + (RC(:,98) * Y(:,41) * Y(:,3)) &
                + (RC(:,42) * Y(:,3) * Y(:,21)) + (RC(:,55) * Y(:,6) * Y(:,32)) 
                L(:) = 0.0 &
                + (RC(:,292)) + (RC(:,293)) &
                + (RC(:,190) * Y(:,5)) + (RC(:,244) * Y(:,9)) + (RC(:,291)) &
                + (RC(:,107) * Y(:,8)) + (RC(:,164) * Y(:,4)) + (RC(:,166) * Y(:,8)) 
                Y(:,22) = (YP(:,22) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H6             Y(:,23) 
                P(:) = 0.0
                L(:) = 0.0 &
                + (RC(:,43) * Y(:,3)) 
                Y(:,23) = (YP(:,23) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H5O2           Y(:,24) 
                P(:) = (DJ(:,64) * Y(:,162)) &
                + (DJ(:,57) * Y(:,148)) + (DJ(:,62) * Y(:,160)) &
                + (DJ(:,38) * Y(:,99)) + (DJ(:,45) * Y(:,127)) &
                + (DJ(:,17) * Y(:,88)) + (DJ(:,33) * Y(:,96)) &
                + (DJ(:,12) * Y(:,71)) + (DJ(:,14) * Y(:,101)) &
                + (RC(:,378) * Y(:,3) * Y(:,96)) + (RC(:,383) * Y(:,3) * Y(:,99)) &
                + (RC(:,303) * Y(:,29)) + (RC(:,323) * Y(:,72)) &
                + (RC(:,194) * Y(:,29) * Y(:,5)) + (RC(:,214) * Y(:,72) * Y(:,5)) &
                + (RC(:,111) * Y(:,29) * Y(:,8)) + (RC(:,131) * Y(:,72) * Y(:,8)) &
                + (RC(:,43) * Y(:,3) * Y(:,23)) + (RC(:,102) * Y(:,3) * Y(:,83)) 
                L(:) = 0.0 &
                + (RC(:,296)) &
                + (RC(:,245) * Y(:,9)) + (RC(:,294)) + (RC(:,295)) &
                + (RC(:,108) * Y(:,8)) + (RC(:,167) * Y(:,8)) + (RC(:,191) * Y(:,5)) 
                Y(:,24) = (YP(:,24) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C3H8             Y(:,25) 
                P(:) = 0.0
                L(:) = 0.0 &
                + (RC(:,44) * Y(:,3)) + (RC(:,45) * Y(:,3)) 
                Y(:,25) = (YP(:,25) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          IC3H7O2          Y(:,26) 
                P(:) = (RC(:,44) * Y(:,3) * Y(:,25)) 
                L(:) = 0.0 &
                + (RC(:,302)) &
                + (RC(:,247) * Y(:,9)) + (RC(:,300)) + (RC(:,301)) &
                + (RC(:,110) * Y(:,8)) + (RC(:,169) * Y(:,8)) + (RC(:,193) * Y(:,5)) 
                Y(:,26) = (YP(:,26) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN10O2           Y(:,27) 
                P(:) = (DJ(:,35) * Y(:,97)) + (DJ(:,65) * Y(:,163)) &
                + (DJ(:,15) * Y(:,186)) + (DJ(:,16) * Y(:,187)) &
                + (RC(:,45) * Y(:,3) * Y(:,25)) + (RC(:,380) * Y(:,3) * Y(:,97)) 
                L(:) = 0.0 &
                + (RC(:,299)) &
                + (RC(:,246) * Y(:,9)) + (RC(:,297)) + (RC(:,298)) &
                + (RC(:,109) * Y(:,8)) + (RC(:,168) * Y(:,8)) + (RC(:,192) * Y(:,5)) 
                Y(:,27) = (YP(:,27) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NC4H10           Y(:,28) 
                P(:) = 0.0
                L(:) = 0.0 &
                + (RC(:,46) * Y(:,3)) 
                Y(:,28) = (YP(:,28) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN13O2           Y(:,29) 
                P(:) = (DJ(:,95) * Y(:,184)) &
                + (DJ(:,37) * Y(:,99)) + (DJ(:,66) * Y(:,164)) &
                + (RC(:,358) * Y(:,3) * Y(:,104)) + (RC(:,382) * Y(:,3) * Y(:,99)) &
                + (RC(:,242) * Y(:,122) * Y(:,5)) + (RC(:,351) * Y(:,122)) &
                + (RC(:,46) * Y(:,3) * Y(:,28)) + (RC(:,163) * Y(:,122) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,303)) + (RC(:,304)) &
                + (RC(:,194) * Y(:,5)) + (RC(:,195) * Y(:,5)) + (RC(:,248) * Y(:,9)) &
                + (RC(:,111) * Y(:,8)) + (RC(:,112) * Y(:,8)) + (RC(:,170) * Y(:,8)) 
                Y(:,29) = (YP(:,29) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H4             Y(:,30) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,54) * Y(:,6)) &
                + (RC(:,47) * Y(:,3)) + (RC(:,50) * Y(:,5)) + (RC(:,53) * Y(:,6)) 
                Y(:,30) = (YP(:,30) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HOCH2CH2O       Y(:,31) 
                P(:) = (RC(:,466) * Y(:,3) * Y(:,192)) &
                + (RC(:,105) * Y(:,3) * Y(:,86)) + (RC(:,106) * Y(:,3) * Y(:,87)) &
                + (RC(:,103) * Y(:,3) * Y(:,84)) + (RC(:,104) * Y(:,3) * Y(:,85)) &
                + (RC(:,47) * Y(:,3) * Y(:,30)) + (RC(:,92) * Y(:,3) * Y(:,77)) 
                L(:) = 0.0 &
                + (RC(:,314)) + (RC(:,315)) &
                + (RC(:,205) * Y(:,5)) + (RC(:,206) * Y(:,5)) + (RC(:,257) * Y(:,9)) &
                + (RC(:,122) *  Y(:,8)) + (RC(:,123) * Y(:,8)) + (RC(:,173) * Y(:,8)) 
                Y(:,31) = (YP(:,31) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C3H6             Y(:,32) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,56) * Y(:,6)) &
                + (RC(:,48) * Y(:,3)) + (RC(:,51) * Y(:,5)) + (RC(:,55) * Y(:,6)) 
                Y(:,32) = (YP(:,32) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN9O2            Y(:,33) 
                P(:) = (RC(:,96) * Y(:,3) * Y(:,79)) + (RC(:,368) * Y(:,3) * Y(:,100)) &
                + (RC(:,48) * Y(:,3) * Y(:,32)) + (RC(:,94) * Y(:,78) * Y(:,3)) 
                L(:) = 0.0 &
                + (RC(:,258) * Y(:,9)) + (RC(:,316)) &
                + (RC(:,124) * Y(:,8)) + (RC(:,174) * Y(:,8)) + (RC(:,207) * Y(:,5)) 
                Y(:,33) = (YP(:,33) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TBUT2ENE         Y(:,34) 
                P(:) = 0.0
                L(:) = 0.0 &
                + (RC(:,58) * Y(:,6)) &
                + (RC(:,49) * Y(:,3)) + (RC(:,52) * Y(:,5)) + (RC(:,57) * Y(:,6)) 
                Y(:,34) = (YP(:,34) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN12O2           Y(:,35) 
                P(:) = (RC(:,369) * Y(:,3) * Y(:,189)) + (RC(:,371) * Y(:,3) * Y(:,191)) &
                + (RC(:,198) * Y(:,93) * Y(:,5)) + (RC(:,305) * Y(:,93)) &
                + (RC(:,49) * Y(:,3) * Y(:,34)) + (RC(:,115) * Y(:,93) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,259) * Y(:,9)) + (RC(:,317)) &
                + (RC(:,125) * Y(:,8)) + (RC(:,175) * Y(:,8)) + (RC(:,208) * Y(:,5)) 
                Y(:,35) = (YP(:,35) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRN6O2           Y(:,36) 
                P(:) = (RC(:,50) * Y(:,5) * Y(:,30)) 
                L(:) = 0.0 &
                + (RC(:,336)) &
                + (RC(:,144) * Y(:,8)) + (RC(:,227) * Y(:,5)) + (RC(:,274) * Y(:,9)) 
                Y(:,36) = (YP(:,36) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRN9O2           Y(:,37) 
                P(:) = (RC(:,51) * Y(:,5) * Y(:,32)) 
                L(:) = 0.0 &
                + (RC(:,337)) &
                + (RC(:,145) * Y(:,8)) + (RC(:,228) * Y(:,5)) + (RC(:,275) * Y(:,9)) 
                Y(:,37) = (YP(:,37) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRN12O2          Y(:,38) 
                P(:) = (RC(:,52) * Y(:,5) * Y(:,34)) 
                L(:) = 0.0 &
                + (RC(:,338)) &
                + (RC(:,146) * Y(:,8)) + (RC(:,229) * Y(:,5)) + (RC(:,276) * Y(:,9)) 
                Y(:,38) = (YP(:,38) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HCHO             Y(:,39) 
                P(:) = (DJ(:,93) * Y(:,182)) + (DJ(:,96) * Y(:,185)) &
                + (DJ(:,80) * Y(:,170)) + (DJ(:,91) * Y(:,180)) &
                + (DJ(:,75) * Y(:,155)) + (DJ(:,79) * Y(:,169) * 2.00) &
                + (DJ(:,74) * Y(:,154) * 2.00) &
                + (DJ(:,63) * Y(:,161)) + (DJ(:,69) * Y(:,166)) &
                + (DJ(:,41) * Y(:,123)) + (DJ(:,53) * Y(:,144)) &
                + (DJ(:,31) * Y(:,115)) + (DJ(:,32) * Y(:,115)) &
                + (DJ(:,22) * Y(:,102)) + (DJ(:,23) * Y(:,46)) &
                + (RC(:,475) * Y(:,3) * Y(:,200)) + (DJ(:,18) * Y(:,111)) &
                + (RC(:,449) * Y(:,3) * Y(:,170)) + (RC(:,473) * Y(:,3) * Y(:,198)) &
                + (RC(:,424) * Y(:,3) * Y(:,144)) &
                + (RC(:,448) * Y(:,3) * Y(:,169) * 2.00) &
                + (RC(:,392) * Y(:,3) * Y(:,123)) + (RC(:,410) * Y(:,3) * Y(:,141)) &
                + (RC(:,362) * Y(:,6) * Y(:,46)) + (RC(:,363) * Y(:,6) * Y(:,46)) &
                + (RC(:,349) * Y(:,54)) + (RC(:,352) * Y(:,55)) &
                + (RC(:,337) * Y(:,37)) + (RC(:,347) * Y(:,119)) &
                + (RC(:,336) * Y(:,36) * 2.00) &
                + (RC(:,334) * Y(:,112)) + (RC(:,335) * Y(:,112)) &
                + (RC(:,325) * Y(:,74)) + (RC(:,330) * Y(:,44)) &
                + (RC(:,316) * Y(:,33)) + (RC(:,324) * Y(:,106)) &
                + (RC(:,314) * Y(:,31) * 2.00) &
                + (RC(:,291) * Y(:,22)) + (RC(:,292) * Y(:,22)) &
                + (RC(:,240) * Y(:,54) * Y(:,5)) + (RC(:,243) * Y(:,55) * Y(:,5)) &
                + (RC(:,228) * Y(:,37) * Y(:,5)) + (RC(:,238) * Y(:,119) * Y(:,5)) &
                + (RC(:,227) * Y(:,36) * Y(:,5)) + (RC(:,227) * Y(:,36) * Y(:,5)) &
                + (RC(:,225) * Y(:,112) * Y(:,5)) + (RC(:,226) * Y(:,112) * Y(:,5)) &
                + (RC(:,216) * Y(:,74) * Y(:,5)) + (RC(:,221) * Y(:,44) * Y(:,5)) &
                + (RC(:,207) * Y(:,33) * Y(:,5)) + (RC(:,215) * Y(:,106) * Y(:,5)) &
                + (RC(:,205) * Y(:,31) * Y(:,5) * 2.00) &
                + (RC(:,162) * Y(:,56) * Y(:,8)) + (RC(:,190) * Y(:,22) * Y(:,5)) &
                + (RC(:,158) * Y(:,54) * Y(:,8)) + (RC(:,160) * Y(:,55) * Y(:,8)) &
                + (RC(:,145) * Y(:,37) * Y(:,8)) + (RC(:,156) * Y(:,119) * Y(:,8)) &
                + (RC(:,144) * Y(:,36) * Y(:,8)) + (RC(:,144) * Y(:,36) * Y(:,8)) &
                + (RC(:,142) * Y(:,112) * Y(:,8)) + (RC(:,143) * Y(:,112) * Y(:,8)) &
                + (RC(:,133) * Y(:,74) * Y(:,8)) + (RC(:,138) * Y(:,44) * Y(:,8)) &
                + (RC(:,124) * Y(:,33) * Y(:,8)) + (RC(:,132) * Y(:,106) * Y(:,8)) &
                + (RC(:,122) * Y(:,31) * Y(:,8) * 2.00) &
                + (RC(:,90) * Y(:,3) * Y(:,76)) + (RC(:,107) * Y(:,22) * Y(:,8)) &
                + (RC(:,71) * Y(:,53) * Y(:,6)) + (RC(:,72) * Y(:,53) * Y(:,6)) &
                + (RC(:,55) * Y(:,6) * Y(:,32)) + (RC(:,56) * Y(:,6) * Y(:,32)) &
                + (RC(:,53) * Y(:,6) * Y(:,30)) + (RC(:,54) * Y(:,6) * Y(:,30)) 
                L(:) = 0.0 &
                + (DJ(:,10)) &
                + (RC(:,82) * Y(:,3)) + (RC(:,85) * Y(:,5)) + (DJ(:,9)) 
                Y(:,39) = (YP(:,39) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HCOOH            Y(:,40) 
                P(:) = (RC(:,74) * Y(:,59) * Y(:,3)) &
                + (RC(:,54) * Y(:,6) * Y(:,30)) + (RC(:,62) * Y(:,6) * Y(:,43)) 
                L(:) = 0.0 &
                + (RC(:,97) * Y(:,3)) 
                Y(:,40) = (YP(:,40) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3CO2H          Y(:,41) 
                P(:) = (RC(:,56) * Y(:,6) * Y(:,32)) + (RC(:,58) * Y(:,6) * Y(:,34)) 
                L(:) = 0.0 &
                + (RC(:,98) * Y(:,3)) 
                Y(:,41) = (YP(:,41) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3CHO           Y(:,42) 
                P(:) = (DJ(:,81) * Y(:,171) * 2.00) &
                + (DJ(:,77) * Y(:,157)) + (DJ(:,80) * Y(:,170)) &
                + (DJ(:,76) * Y(:,156) * 2.00) &
                + (DJ(:,57) * Y(:,148)) + (DJ(:,75) * Y(:,155)) &
                + (DJ(:,45) * Y(:,127)) + (DJ(:,54) * Y(:,145)) &
                + (DJ(:,20) * Y(:,104)) + (DJ(:,42) * Y(:,124)) &
                + (RC(:,474) * Y(:,3) * Y(:,199)) + (DJ(:,19) * Y(:,188)) &
                + (RC(:,449) * Y(:,3) * Y(:,170)) &
                + (RC(:,450) * Y(:,3) * Y(:,171) * 2.00) &
                + (RC(:,393) * Y(:,3) * Y(:,124)) + (RC(:,425) * Y(:,3) * Y(:,145)) &
                + (RC(:,338) * Y(:,38) * 2.00) &
                + (RC(:,327) * Y(:,107)) + (RC(:,337) * Y(:,37)) &
                + (RC(:,318) * Y(:,95)) + (RC(:,326) * Y(:,75)) &
                + (RC(:,317) * Y(:,35) * 2.00) &
                + (RC(:,303) * Y(:,29)) + (RC(:,316) * Y(:,33)) &
                + (RC(:,294) * Y(:,24)) + (RC(:,295) * Y(:,24)) &
                + (RC(:,229) * Y(:,38) * Y(:,5) * 2.00) &
                + (RC(:,218) * Y(:,107) * Y(:,5)) + (RC(:,228) * Y(:,37) * Y(:,5)) &
                + (RC(:,209) * Y(:,95) * Y(:,5)) + (RC(:,217) * Y(:,75) * Y(:,5)) &
                + (RC(:,207) * Y(:,33) * Y(:,5)) + (RC(:,208) * Y(:,35) * Y(:,5) * 2.00) &
                + (RC(:,191) * Y(:,24) * Y(:,5)) + (RC(:,194) * Y(:,29) * Y(:,5)) &
                + (RC(:,146) * Y(:,38) * Y(:,8) * 2.00) &
                + (RC(:,135) * Y(:,107) * Y(:,8)) + (RC(:,145) * Y(:,37) * Y(:,8)) &
                + (RC(:,126) * Y(:,95) * Y(:,8)) + (RC(:,134) * Y(:,75) * Y(:,8)) &
                + (RC(:,125) * Y(:,35) * Y(:,8) * 2.00) &
                + (RC(:,111) * Y(:,29) * Y(:,8)) + (RC(:,124) * Y(:,33) * Y(:,8)) &
                + (RC(:,91) * Y(:,3) * Y(:,77)) + (RC(:,108) * Y(:,24) * Y(:,8)) &
                + (RC(:,57) * Y(:,6) * Y(:,34)) + (RC(:,58) * Y(:,6) * Y(:,34)) 
                L(:) = 0.0 &
                + (RC(:,83) * Y(:,3)) + (RC(:,86) * Y(:,5)) + (DJ(:,11)) 
                Y(:,42) = (YP(:,42) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C5H8             Y(:,43) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,62) * Y(:,6)) &
                + (RC(:,59) * Y(:,3)) + (RC(:,60) * Y(:,5)) + (RC(:,61) * Y(:,6)) 
                Y(:,43) = (YP(:,43) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RU14O2           Y(:,44) 
                P(:) = (RC(:,59) * Y(:,3) * Y(:,43)) 
                L(:) = 0.0 &
                + (RC(:,329)) + (RC(:,330)) &
                + (RC(:,220) * Y(:,5)) + (RC(:,221) * Y(:,5)) + (RC(:,271) * Y(:,9)) &
                + (RC(:,137) * Y(:,8)) + (RC(:,138) * Y(:,8)) + (RC(:,180) * Y(:,8)) 
                Y(:,44) = (YP(:,44) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRU14O2          Y(:,45) 
                P(:) = (RC(:,60) * Y(:,5) * Y(:,43)) 
                L(:) = 0.0 &
                + (RC(:,339)) &
                + (RC(:,147) * Y(:,8)) + (RC(:,230) * Y(:,5)) + (RC(:,277) * Y(:,9)) 
                Y(:,45) = (YP(:,45) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          UCARB10          Y(:,46) 
                P(:) = (DJ(:,69) * Y(:,166)) &
                + (RC(:,330) * Y(:,44)) + (RC(:,481) * Y(:,3) * Y(:,201)) &
                + (RC(:,138) * Y(:,44) * Y(:,8)) + (RC(:,221) * Y(:,44) * Y(:,5)) &
                + (RC(:,61) * Y(:,6) * Y(:,43)) + (RC(:,62) * Y(:,6) * Y(:,43)) 
                L(:) = 0.0 &
                + (RC(:,363) * Y(:,6)) + (DJ(:,23)) &
                + (RC(:,360) * Y(:,3)) + (RC(:,361) * Y(:,5)) + (RC(:,362) * Y(:,6)) 
                Y(:,46) = (YP(:,46) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          APINENE          Y(:,47) 
                P(:) = 0.0
                L(:) = 0.0 &
                + (RC(:,66) * Y(:,6)) + (RC(:,67) * Y(:,6)) &
                + (RC(:,63) * Y(:,3)) + (RC(:,64) * Y(:,5)) + (RC(:,65) * Y(:,6)) 
                Y(:,47) = (YP(:,47) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN28O2          Y(:,48) 
                P(:) = (RC(:,63) * Y(:,47) * Y(:,3)) 
                L(:) = 0.0 &
                + (RC(:,232) * Y(:,5)) + (RC(:,279) * Y(:,9)) + (RC(:,341)) &
                + (RC(:,149) * Y(:,8)) + (RC(:,150) * Y(:,8)) + (RC(:,185) * Y(:,8)) 
                Y(:,48) = (YP(:,48) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRTN28O2         Y(:,49) 
                P(:) = (RC(:,64) * Y(:,47) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,342)) &
                + (RC(:,151) * Y(:,8)) + (RC(:,233) * Y(:,5)) + (RC(:,280) * Y(:,9)) 
                Y(:,49) = (YP(:,49) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN26O2          Y(:,50) 
                P(:) = (RC(:,483) * Y(:,203)) + (DJ(:,39) * Y(:,51)) &
                + (RC(:,387) * Y(:,5) * Y(:,51)) + (RC(:,455) * Y(:,3) * Y(:,176)) &
                + (RC(:,65) * Y(:,47) * Y(:,6)) + (RC(:,384) * Y(:,3) * Y(:,51)) 
                L(:) = 0.0 &
                + (RC(:,343)) + (RC(:,482) * Y(:,4)) &
                + (RC(:,152) * Y(:,8)) + (RC(:,234) * Y(:,5)) + (RC(:,281) * Y(:,9)) 
                Y(:,50) = (YP(:,50) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TNCARB26         Y(:,51) 
                P(:) = (DJ(:,85) * Y(:,174)) + (DJ(:,86) * Y(:,175)) &
                + (RC(:,454) * Y(:,3) * Y(:,174)) + (RC(:,456) * Y(:,3) * Y(:,175)) &
                + (RC(:,342) * Y(:,49)) + (RC(:,408) * Y(:,3) * Y(:,139)) &
                + (RC(:,233) * Y(:,49) * Y(:,5)) + (RC(:,341) * Y(:,48)) &
                + (RC(:,151) * Y(:,49) * Y(:,8)) + (RC(:,232) * Y(:,48) * Y(:,5)) &
                + (RC(:,66) * Y(:,47) * Y(:,6)) + (RC(:,149) * Y(:,48) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,384) * Y(:,3)) + (RC(:,387) * Y(:,5)) + (DJ(:,39)) 
                Y(:,51) = (YP(:,51) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RCOOH25          Y(:,52) 
                P(:) = (RC(:,67) * Y(:,47) * Y(:,6)) + (RC(:,490) * Y(:,206)) 
                L(:) = 0.0 &
                + (RC(:,389) * Y(:,3)) + (RC(:,489)) 
                Y(:,52) = (YP(:,52) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          BPINENE          Y(:,53) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,71) * Y(:,6)) + (RC(:,72) * Y(:,6)) + (RC(:,73) * Y(:,6)) &
                + (RC(:,68) * Y(:,3)) + (RC(:,69) * Y(:,5)) + (RC(:,70) * Y(:,6)) 
                Y(:,53) = (YP(:,53) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTX28O2          Y(:,54) 
                P(:) = (RC(:,68) * Y(:,53) * Y(:,3)) + (RC(:,462) * Y(:,3) * Y(:,182)) 
                L(:) = 0.0 &
                + (RC(:,240) * Y(:,5)) + (RC(:,287) * Y(:,9)) + (RC(:,349)) &
                + (RC(:,158) * Y(:,8)) + (RC(:,159) * Y(:,8)) + (RC(:,187) * Y(:,8)) 
                Y(:,54) = (YP(:,54) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRTX28O2         Y(:,55) 
                P(:) = (RC(:,69) * Y(:,53) * Y(:,5)) + (RC(:,465) * Y(:,3) * Y(:,185)) 
                L(:) = 0.0 &
                + (RC(:,352)) &
                + (RC(:,160) * Y(:,8)) + (RC(:,243) * Y(:,5)) + (RC(:,290) * Y(:,9)) 
                Y(:,55) = (YP(:,55) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTX24O2          Y(:,56) 
                P(:) = (RC(:,70) * Y(:,53) * Y(:,6)) + (RC(:,390) * Y(:,3) * Y(:,57)) 
                L(:) = 0.0 &
                + (RC(:,241) * Y(:,5)) + (RC(:,288) * Y(:,9)) + (RC(:,350)) &
                + (RC(:,161) * Y(:,8)) + (RC(:,162) * Y(:,8)) + (RC(:,188) * Y(:,8)) 
                Y(:,56) = (YP(:,56) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TXCARB24         Y(:,57) 
                P(:) = (DJ(:,96) * Y(:,185)) &
                + (RC(:,410) * Y(:,3) * Y(:,141)) + (DJ(:,93) * Y(:,182)) &
                + (RC(:,349) * Y(:,54)) + (RC(:,352) * Y(:,55)) &
                + (RC(:,240) * Y(:,54) * Y(:,5)) + (RC(:,243) * Y(:,55) * Y(:,5)) &
                + (RC(:,158) * Y(:,54) * Y(:,8)) + (RC(:,160) * Y(:,55) * Y(:,8)) &
                + (RC(:,71) * Y(:,53) * Y(:,6)) + (RC(:,73) * Y(:,53) * Y(:,6)) 
                L(:) = 0.0 &
                + (RC(:,390) * Y(:,3)) 
                Y(:,57) = (YP(:,57) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TXCARB22         Y(:,58) 
                P(:) = (DJ(:,52) * Y(:,142)) + (DJ(:,94) * Y(:,183)) &
                + (RC(:,411) * Y(:,3) * Y(:,142)) + (RC(:,463) * Y(:,3) * Y(:,183)) &
                + (RC(:,241) * Y(:,56) * Y(:,5)) + (RC(:,350) * Y(:,56)) &
                + (RC(:,72) * Y(:,53) * Y(:,6)) + (RC(:,161) * Y(:,56) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,391) * Y(:,3)) 
                Y(:,58) = (YP(:,58) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H2             Y(:,59) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,74) * Y(:,3)) + (RC(:,75) * Y(:,3)) 
                Y(:,59) = (YP(:,59) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB3            Y(:,60) 
                P(:) = (DJ(:,83) * Y(:,152)) &
                + (DJ(:,50) * Y(:,137)) + (DJ(:,82) * Y(:,151)) &
                + (RC(:,452) * Y(:,3) * Y(:,152)) + (DJ(:,49) * Y(:,136)) &
                + (RC(:,406) * Y(:,3) * Y(:,137)) + (RC(:,451) * Y(:,3) * Y(:,151)) &
                + (RC(:,311) * Y(:,69)) + (RC(:,405) * Y(:,3) * Y(:,136)) &
                + (RC(:,308) * Y(:,65)) + (RC(:,310) * Y(:,68)) &
                + (RC(:,203) * Y(:,68) * Y(:,5)) + (RC(:,307) * Y(:,62)) &
                + (RC(:,200) * Y(:,62) * Y(:,5)) + (RC(:,201) * Y(:,65) * Y(:,5)) &
                + (RC(:,118) * Y(:,65) * Y(:,8)) + (RC(:,120) * Y(:,68) * Y(:,8)) &
                + (RC(:,75) * Y(:,59) * Y(:,3)) + (RC(:,117) * Y(:,62) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,366) * Y(:,3)) + (DJ(:,24)) 
                Y(:,60) = (YP(:,60) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          BENZENE          Y(:,61) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,76) * Y(:,3)) + (RC(:,77) * Y(:,3)) 
                Y(:,61) = (YP(:,61) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA13O2           Y(:,62) 
                P(:) = (RC(:,76) * Y(:,61) * Y(:,3)) 
                L(:) = 0.0 &
                + (RC(:,253) * Y(:,9)) + (RC(:,307)) &
                + (RC(:,117) * Y(:,8)) + (RC(:,181) * Y(:,8)) + (RC(:,200) * Y(:,5)) 
                Y(:,62) = (YP(:,62) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          AROH14           Y(:,63) 
                P(:) = (RC(:,77) * Y(:,61) * Y(:,3)) 
                L(:) = 0.0 &
                + (RC(:,413) * Y(:,3)) + (RC(:,414) * Y(:,5)) 
                Y(:,63) = (YP(:,63) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TOLUENE          Y(:,64) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,78) * Y(:,3)) + (RC(:,79) * Y(:,3)) 
                Y(:,64) = (YP(:,64) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA16O2           Y(:,65) 
                P(:) = (RC(:,78) * Y(:,64) * Y(:,3)) 
                L(:) = 0.0 &
                + (RC(:,308)) + (RC(:,309)) &
                + (RC(:,201) * Y(:,5)) + (RC(:,202) * Y(:,5)) + (RC(:,254) * Y(:,9)) &
                + (RC(:,118) * Y(:,8)) + (RC(:,119) * Y(:,8)) + (RC(:,182) * Y(:,8)) 
                Y(:,65) = (YP(:,65) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          AROH17           Y(:,66) 
                P(:) = (RC(:,79) * Y(:,64) * Y(:,3)) 
                L(:) = 0.0 &
                + (RC(:,418) * Y(:,3)) + (RC(:,419) * Y(:,5)) 
                Y(:,66) = (YP(:,66) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          OXYL             Y(:,67) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,80) * Y(:,3)) + (RC(:,81) * Y(:,3)) 
                Y(:,67) = (YP(:,67) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA19AO2          Y(:,68) 
                P(:) = (RC(:,80) * Y(:,67) * Y(:,3)) 
                L(:) = 0.0 &
                + (RC(:,255) * Y(:,9)) + (RC(:,310)) &
                + (RC(:,120) * Y(:,8)) + (RC(:,183) * Y(:,8)) + (RC(:,203) * Y(:,5)) 
                Y(:,68) = (YP(:,68) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA19CO2          Y(:,69) 
                P(:) = (RC(:,81) * Y(:,67) * Y(:,3)) 
                L(:) = 0.0 &
                + (RC(:,256) * Y(:,9)) + (RC(:,311)) &
                + (RC(:,121) * Y(:,8)) + (RC(:,184) * Y(:,8)) + (RC(:,204) * Y(:,5)) 
                Y(:,69) = (YP(:,69) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3CO3           Y(:,70) 
                P(:) = (DJ(:,71) * Y(:,168)) &
                + (DJ(:,40) * Y(:,120) * 2.00) &
                + (DJ(:,31) * Y(:,115)) + (DJ(:,32) * Y(:,115)) &
                + (DJ(:,27) * Y(:,189)) + (DJ(:,29) * Y(:,109)) &
                + (DJ(:,25) * Y(:,98)) + (DJ(:,26) * Y(:,100) * 2.00) &
                + (DJ(:,19) * Y(:,188)) + (DJ(:,23) * Y(:,46)) &
                + (DJ(:,17) * Y(:,88)) + (DJ(:,18) * Y(:,111)) &
                + (DJ(:,14) * Y(:,101)) + (DJ(:,15) * Y(:,186)) &
                + (RC(:,468) * Y(:,198)) + (DJ(:,13) * Y(:,73)) &
                + (RC(:,374) * Y(:,6) * Y(:,109)) + (RC(:,431) * Y(:,3) * Y(:,159)) &
                + (RC(:,362) * Y(:,6) * Y(:,46)) + (RC(:,367) * Y(:,3) * Y(:,98)) &
                + (RC(:,331) * Y(:,110)) + (RC(:,333) * Y(:,112)) &
                + (RC(:,325) * Y(:,74)) + (RC(:,326) * Y(:,75)) &
                + (RC(:,222) * Y(:,110) * Y(:,5)) + (RC(:,224) * Y(:,112) * Y(:,5)) &
                + (RC(:,216) * Y(:,74) * Y(:,5)) + (RC(:,217) * Y(:,75) * Y(:,5)) &
                + (RC(:,139) * Y(:,110) * Y(:,8)) + (RC(:,141) * Y(:,112) * Y(:,8)) &
                + (RC(:,133) * Y(:,74) * Y(:,8)) + (RC(:,134) * Y(:,75) * Y(:,8)) &
                + (RC(:,83) * Y(:,3) * Y(:,42)) + (RC(:,86) * Y(:,5) * Y(:,42)) 
                L(:) = 0.0 &
                + (RC(:,322)) + (RC(:,467) * Y(:,4)) &
                + (RC(:,130) * Y(:,8)) + (RC(:,213) * Y(:,5)) + (RC(:,264) * Y(:,9)) 
                Y(:,70) = (YP(:,70) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H5CHO          Y(:,71) 
                P(:) = (DJ(:,78) * Y(:,158) * 2.00) &
                + (DJ(:,55) * Y(:,146)) + (DJ(:,77) * Y(:,157)) &
                + (DJ(:,21) * Y(:,105)) + (DJ(:,43) * Y(:,125)) &
                + (RC(:,394) * Y(:,3) * Y(:,125)) + (RC(:,426) * Y(:,3) * Y(:,146)) &
                + (RC(:,318) * Y(:,95)) + (RC(:,319) * Y(:,103) * 2.00) &
                + (RC(:,297) * Y(:,27)) + (RC(:,298) * Y(:,27)) &
                + (RC(:,210) * Y(:,103) * Y(:,5) * 2.00) &
                + (RC(:,192) * Y(:,27) * Y(:,5)) + (RC(:,209) * Y(:,95) * Y(:,5)) &
                + (RC(:,126) * Y(:,95) * Y(:,8)) &
                + (RC(:,127) * Y(:,103) * Y(:,8) * 2.00) &
                + (RC(:,93) * Y(:,78) * Y(:,3)) + (RC(:,109) * Y(:,27) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,84) * Y(:,3)) + (RC(:,87) * Y(:,5)) + (DJ(:,12)) 
                Y(:,71) = (YP(:,71) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H5CO3          Y(:,72) 
                P(:) = (RC(:,470) * Y(:,199)) &
                + (RC(:,327) * Y(:,107)) + (RC(:,432) * Y(:,3) * Y(:,160)) &
                + (RC(:,135) * Y(:,107) * Y(:,8)) + (RC(:,218) * Y(:,107) * Y(:,5)) &
                + (RC(:,84) * Y(:,3) * Y(:,71)) + (RC(:,87) * Y(:,5) * Y(:,71)) 
                L(:) = 0.0 &
                + (RC(:,323)) + (RC(:,469) * Y(:,4)) &
                + (RC(:,131) * Y(:,8)) + (RC(:,214) * Y(:,5)) + (RC(:,265) * Y(:,9)) 
                Y(:,72) = (YP(:,72) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3COCH3         Y(:,73) 
                P(:) = (DJ(:,90) * Y(:,179)) + (DJ(:,95) * Y(:,184)) &
                + (DJ(:,44) * Y(:,126)) + (DJ(:,56) * Y(:,147)) &
                + (RC(:,464) * Y(:,3) * Y(:,184)) + (RC(:,484) * Y(:,3) * Y(:,203)) &
                + (RC(:,412) * Y(:,3) * Y(:,143)) + (RC(:,427) * Y(:,3) * Y(:,147)) &
                + (RC(:,395) * Y(:,3) * Y(:,126)) + (RC(:,409) * Y(:,3) * Y(:,140)) &
                + (RC(:,346) * Y(:,118)) + (RC(:,351) * Y(:,122)) &
                + (RC(:,300) * Y(:,26)) + (RC(:,301) * Y(:,26)) &
                + (RC(:,237) * Y(:,118) * Y(:,5)) + (RC(:,242) * Y(:,122) * Y(:,5)) &
                + (RC(:,163) * Y(:,122) * Y(:,8)) + (RC(:,193) * Y(:,26) * Y(:,5)) &
                + (RC(:,159) * Y(:,54) * Y(:,8)) + (RC(:,162) * Y(:,56) * Y(:,8)) &
                + (RC(:,150) * Y(:,48) * Y(:,8)) + (RC(:,155) * Y(:,118) * Y(:,8)) &
                + (RC(:,95) * Y(:,3) * Y(:,79)) + (RC(:,110) * Y(:,26) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,88) * Y(:,3)) + (DJ(:,13)) 
                Y(:,73) = (YP(:,73) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN8O2            Y(:,74) 
                P(:) = (DJ(:,92) * Y(:,181)) &
                + (DJ(:,28) * Y(:,190) * 2.00) &
                + (DJ(:,21) * Y(:,105)) + (DJ(:,27) * Y(:,189)) &
                + (DJ(:,16) * Y(:,187)) + (DJ(:,20) * Y(:,104)) &
                + (RC(:,239) * Y(:,121) * Y(:,5)) + (RC(:,348) * Y(:,121)) &
                + (RC(:,88) * Y(:,3) * Y(:,73)) + (RC(:,157) * Y(:,121) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,325)) &
                + (RC(:,133) * Y(:,8)) + (RC(:,216) * Y(:,5)) + (RC(:,267) * Y(:,9)) 
                Y(:,74) = (YP(:,74) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN11O2           Y(:,75) 
                P(:) = (RC(:,89) * Y(:,101) * Y(:,3)) + (RC(:,355) * Y(:,3) * Y(:,88)) 
                L(:) = 0.0 &
                + (RC(:,326)) &
                + (RC(:,134) * Y(:,8)) + (RC(:,217) * Y(:,5)) + (RC(:,268) * Y(:,9)) 
                Y(:,75) = (YP(:,75) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3OH            Y(:,76) 
                P(:) = (RC(:,293) * Y(:,22)) 
                L(:) = 0.0 &
                + (RC(:,90) * Y(:,3)) 
                Y(:,76) = (YP(:,76) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H5OH           Y(:,77) 
                P(:) = (RC(:,296) * Y(:,24)) 
                L(:) = 0.0 &
                + (RC(:,91) * Y(:,3)) + (RC(:,92) * Y(:,3)) 
                Y(:,77) = (YP(:,77) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NPROPOL          Y(:,78) 
                P(:) = (RC(:,299) * Y(:,27)) 
                L(:) = 0.0 &
                + (RC(:,93) * Y(:,3)) + (RC(:,94) * Y(:,3)) 
                Y(:,78) = (YP(:,78) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          IPROPOL          Y(:,79) 
                P(:) = (RC(:,302) * Y(:,26)) 
                L(:) = 0.0 &
                + (RC(:,95) * Y(:,3)) + (RC(:,96) * Y(:,3)) 
                Y(:,79) = (YP(:,79) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3CL            Y(:,80) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,99) * Y(:,3)) 
                Y(:,80) = (YP(:,80) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH2CL2           Y(:,81) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,100) * Y(:,3)) 
                Y(:,81) = (YP(:,81) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CHCL3            Y(:,82) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,101) * Y(:,3)) 
                Y(:,82) = (YP(:,82) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3CCL3          Y(:,83) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,102) * Y(:,3)) 
                Y(:,83) = (YP(:,83) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TCE              Y(:,84) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,103) * Y(:,3)) 
                Y(:,84) = (YP(:,84) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TRICLETH         Y(:,85) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,104) * Y(:,3)) 
                Y(:,85) = (YP(:,85) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CDICLETH         Y(:,86) 
                P(:) = 0.0
                L(:) = 0.0 &
                + (RC(:,105) * Y(:,3)) 
                Y(:,86) = (YP(:,86) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TDICLETH         Y(:,87) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,106) * Y(:,3)) 
                Y(:,87) = (YP(:,87) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB11A          Y(:,88) 
                P(:) = (DJ(:,58) * Y(:,148)) &
                + (RC(:,428) * Y(:,3) * Y(:,148)) + (DJ(:,46) * Y(:,127)) &
                + (RC(:,304) * Y(:,29)) + (RC(:,396) * Y(:,3) * Y(:,127)) &
                + (RC(:,112) * Y(:,29) * Y(:,8)) + (RC(:,195) * Y(:,29) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,355) * Y(:,3)) + (DJ(:,17)) 
                Y(:,88) = (YP(:,88) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN16O2           Y(:,89) 
                P(:) = (RC(:,359) * Y(:,3) * Y(:,105)) + (DJ(:,67) * Y(:,165)) 
                L(:) = 0.0 &
                + (RC(:,249) * Y(:,9)) + (RC(:,312)) &
                + (RC(:,113) * Y(:,8)) + (RC(:,171) * Y(:,8)) + (RC(:,196) * Y(:,5)) 
                Y(:,89) = (YP(:,89) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN15AO2          Y(:,90) 
                P(:) = (DJ(:,59) * Y(:,149)) &
                + (RC(:,312) * Y(:,89)) + (RC(:,385) * Y(:,3) * Y(:,193)) &
                + (RC(:,113) * Y(:,89) * Y(:,8)) + (RC(:,196) * Y(:,89) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,262) * Y(:,9)) + (RC(:,320)) &
                + (RC(:,128) * Y(:,8)) + (RC(:,178) * Y(:,8)) + (RC(:,211) * Y(:,5)) 
                Y(:,90) = (YP(:,90) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN19O2           Y(:,91) 
                P(:) = (RC(:,150) * Y(:,48) * Y(:,8)) + (RC(:,159) * Y(:,54) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,250) * Y(:,9)) + (RC(:,313)) &
                + (RC(:,114) * Y(:,8)) + (RC(:,172) * Y(:,8)) + (RC(:,197) * Y(:,5)) 
                Y(:,91) = (YP(:,91) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN18AO2          Y(:,92) 
                P(:) = (RC(:,313) * Y(:,91)) + (DJ(:,60) * Y(:,150)) &
                + (RC(:,114) * Y(:,91) * Y(:,8)) + (RC(:,197) * Y(:,91) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,263) * Y(:,9)) + (RC(:,321)) &
                + (RC(:,129) * Y(:,8)) + (RC(:,179) * Y(:,8)) + (RC(:,212) * Y(:,5)) 
                Y(:,92) = (YP(:,92) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN13AO2          Y(:,93) 
                P(:) = (RC(:,162) * Y(:,56) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,305)) &
                + (RC(:,115) * Y(:,8)) + (RC(:,198) * Y(:,5)) + (RC(:,251) * Y(:,9)) 
                Y(:,93) = (YP(:,93) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN16AO2          Y(:,94) 
                P(:) = (RC(:,328) * Y(:,108)) &
                + (RC(:,136) * Y(:,108) * Y(:,8)) + (RC(:,219) * Y(:,108) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,306)) &
                + (RC(:,116) * Y(:,8)) + (RC(:,199) * Y(:,5)) + (RC(:,252) * Y(:,9)) 
                Y(:,94) = (YP(:,94) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN15O2           Y(:,95) 
                P(:) = (DJ(:,47) * Y(:,128)) &
                + (RC(:,306) * Y(:,94)) + (RC(:,370) * Y(:,3) * Y(:,190)) &
                + (RC(:,116) * Y(:,94) * Y(:,8)) + (RC(:,199) * Y(:,94) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,260) * Y(:,9)) + (RC(:,318)) &
                + (RC(:,126) * Y(:,8)) + (RC(:,176) * Y(:,8)) + (RC(:,209) * Y(:,5)) 
                Y(:,95) = (YP(:,95) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          UDCARB8          Y(:,96) 
                P(:) = (DJ(:,49) * Y(:,136)) + (DJ(:,82) * Y(:,151)) &
                + (RC(:,405) * Y(:,3) * Y(:,136)) + (RC(:,451) * Y(:,3) * Y(:,151)) &
                + (RC(:,307) * Y(:,62)) + (RC(:,309) * Y(:,65)) &
                + (RC(:,202) * Y(:,65) * Y(:,5)) + (RC(:,204) * Y(:,69) * Y(:,5)) &
                + (RC(:,121) * Y(:,69) * Y(:,8)) + (RC(:,200) * Y(:,62) * Y(:,5)) &
                + (RC(:,117) * Y(:,62) * Y(:,8)) + (RC(:,119) * Y(:,65) * Y(:,8)) 
                L(:) = 0.0 &
                + (DJ(:,34)) &
                + (RC(:,378) * Y(:,3)) + (RC(:,379) * Y(:,3)) + (DJ(:,33)) 
                Y(:,96) = (YP(:,96) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          UDCARB11         Y(:,97) 
                P(:) = (DJ(:,84) * Y(:,153)) &
                + (DJ(:,51) * Y(:,138)) + (DJ(:,83) * Y(:,152)) &
                + (RC(:,453) * Y(:,3) * Y(:,153)) + (DJ(:,50) * Y(:,137)) &
                + (RC(:,407) * Y(:,3) * Y(:,138)) + (RC(:,452) * Y(:,3) * Y(:,152)) &
                + (RC(:,308) * Y(:,65)) + (RC(:,406) * Y(:,3) * Y(:,137)) &
                + (RC(:,118) * Y(:,65) * Y(:,8)) + (RC(:,201) * Y(:,65) * Y(:,5)) 
                L(:) = 0.0 &
                + (DJ(:,36)) &
                + (RC(:,380) * Y(:,3)) + (RC(:,381) * Y(:,3)) + (DJ(:,35)) 
                Y(:,97) = (YP(:,97) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB6            Y(:,98) 
                P(:) = (DJ(:,70) * Y(:,167)) + (DJ(:,84) * Y(:,153)) &
                + (RC(:,453) * Y(:,3) * Y(:,153)) + (DJ(:,51) * Y(:,138)) &
                + (RC(:,407) * Y(:,3) * Y(:,138)) + (RC(:,434) * Y(:,3) * Y(:,162)) &
                + (RC(:,375) * Y(:,6) * Y(:,109)) + (RC(:,377) * Y(:,3) * Y(:,115)) &
                + (RC(:,356) * Y(:,3) * Y(:,111)) + (RC(:,363) * Y(:,6) * Y(:,46)) &
                + (RC(:,309) * Y(:,65)) + (RC(:,334) * Y(:,112)) &
                + (RC(:,202) * Y(:,65) * Y(:,5)) + (RC(:,225) * Y(:,112) * Y(:,5)) &
                + (RC(:,119) * Y(:,65) * Y(:,8)) + (RC(:,142) * Y(:,112) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,367) * Y(:,3)) + (DJ(:,25)) 
                Y(:,98) = (YP(:,98) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          UDCARB14         Y(:,99) 
                P(:) = (RC(:,310) * Y(:,68)) + (RC(:,311) * Y(:,69)) &
                + (RC(:,120) * Y(:,68) * Y(:,8)) + (RC(:,203) * Y(:,68) * Y(:,5)) 
                L(:) = 0.0 &
                + (DJ(:,38)) &
                + (RC(:,382) * Y(:,3)) + (RC(:,383) * Y(:,3)) + (DJ(:,37)) 
                Y(:,99) = (YP(:,99) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB9            Y(:,100) 
                P(:) = (RC(:,357) * Y(:,3) * Y(:,188)) + (RC(:,435) * Y(:,3) * Y(:,163)) &
                + (RC(:,121) * Y(:,69) * Y(:,8)) + (RC(:,204) * Y(:,69) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,368) * Y(:,3)) + (DJ(:,26)) 
                Y(:,100) = (YP(:,100) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          MEK              Y(:,101) 
                P(:) = 0.0 
                L(:) = 0.0 &
                + (RC(:,89) * Y(:,3)) + (DJ(:,14)) 
                Y(:,101) = (YP(:,101) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HOCH2CHO         Y(:,102) 
                P(:) = (DJ(:,71) * Y(:,168)) &
                + (DJ(:,29) * Y(:,109)) + (DJ(:,70) * Y(:,167)) &
                + (RC(:,399) * Y(:,3) * Y(:,130)) + (RC(:,443) * Y(:,3) * Y(:,154)) &
                + (RC(:,374) * Y(:,6) * Y(:,109)) + (RC(:,375) * Y(:,6) * Y(:,109)) &
                + (RC(:,332) * Y(:,110)) + (RC(:,333) * Y(:,112)) &
                + (RC(:,315) * Y(:,31)) + (RC(:,331) * Y(:,110)) &
                + (RC(:,222) * Y(:,110) * Y(:,5)) + (RC(:,224) * Y(:,112) * Y(:,5)) &
                + (RC(:,141) * Y(:,112) * Y(:,8)) + (RC(:,206) * Y(:,31) * Y(:,5)) &
                + (RC(:,123) * Y(:,31) * Y(:,8)) + (RC(:,139) * Y(:,110) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,364) * Y(:,3)) + (RC(:,365) * Y(:,5)) + (DJ(:,22)) 
                Y(:,102) = (YP(:,102) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN18O2           Y(:,103) 
                P(:) = (DJ(:,48) * Y(:,129)) 
                L(:) = 0.0 &
                + (RC(:,261) * Y(:,9)) + (RC(:,319)) &
                + (RC(:,127) * Y(:,8)) + (RC(:,177) * Y(:,8)) + (RC(:,210) * Y(:,5)) 
                Y(:,103) = (YP(:,103) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB13           Y(:,104) 
                P(:) = (RC(:,446) * Y(:,3) * Y(:,157)) &
                + (RC(:,416) * Y(:,3) * Y(:,195)) + (RC(:,417) * Y(:,5) * Y(:,195)) &
                + (RC(:,320) * Y(:,90)) + (RC(:,402) * Y(:,3) * Y(:,133)) &
                + (RC(:,128) * Y(:,90) * Y(:,8)) + (RC(:,211) * Y(:,90) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,358) * Y(:,3)) + (DJ(:,20)) 
                Y(:,104) = (YP(:,104) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB16           Y(:,105) 
                P(:) = (RC(:,447) * Y(:,3) * Y(:,158)) + (RC(:,484) * Y(:,3) * Y(:,203)) &
                + (RC(:,421) * Y(:,3) * Y(:,197)) + (RC(:,422) * Y(:,5) * Y(:,197)) &
                + (RC(:,321) * Y(:,92)) + (RC(:,403) * Y(:,3) * Y(:,134)) &
                + (RC(:,129) * Y(:,92) * Y(:,8)) + (RC(:,212) * Y(:,92) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,359) * Y(:,3)) + (DJ(:,21)) 
                Y(:,105) = (YP(:,105) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HOCH2CO3         Y(:,106) 
                P(:) = (RC(:,433) * Y(:,3) * Y(:,161)) + (RC(:,472) * Y(:,200)) &
                + (RC(:,364) * Y(:,3) * Y(:,102)) + (RC(:,365) * Y(:,5) * Y(:,102)) 
                L(:) = 0.0 &
                + (RC(:,324)) + (RC(:,471) * Y(:,4)) &
                + (RC(:,132) * Y(:,8)) + (RC(:,215) * Y(:,5)) + (RC(:,266) * Y(:,9)) 
                Y(:,106) = (YP(:,106) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN14O2           Y(:,107) 
                P(:) = (RC(:,353) * Y(:,3) * Y(:,186)) 
                L(:) = 0.0 &
                + (RC(:,327)) &
                + (RC(:,135) * Y(:,8)) + (RC(:,218) * Y(:,5)) + (RC(:,269) * Y(:,9)) 
                Y(:,107) = (YP(:,107) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN17O2           Y(:,108) 
                P(:) = (RC(:,354) * Y(:,3) * Y(:,187)) 
                L(:) = 0.0 &
                + (RC(:,328)) &
                + (RC(:,136) * Y(:,8)) + (RC(:,219) * Y(:,5)) + (RC(:,270) * Y(:,9)) 
                Y(:,108) = (YP(:,108) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          UCARB12          Y(:,109) 
                P(:) = (RC(:,438) * Y(:,3) * Y(:,166)) + (DJ(:,68) * Y(:,166)) &
                + (RC(:,329) * Y(:,44)) + (RC(:,404) * Y(:,3) * Y(:,135)) &
                + (RC(:,137) * Y(:,44) * Y(:,8)) + (RC(:,220) * Y(:,44) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,375) * Y(:,6)) + (DJ(:,29)) &
                + (RC(:,372) * Y(:,3)) + (RC(:,373) * Y(:,5)) + (RC(:,374) * Y(:,6)) 
                Y(:,109) = (YP(:,109) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RU12O2           Y(:,110) 
                P(:) = (RC(:,439) * Y(:,3) * Y(:,167)) + (RC(:,477) * Y(:,201)) &
                + (RC(:,372) * Y(:,3) * Y(:,109)) + (RC(:,373) * Y(:,5) * Y(:,109)) 
                L(:) = 0.0 &
                + (RC(:,332)) + (RC(:,476) * Y(:,4)) &
                + (RC(:,223) * Y(:,5)) + (RC(:,272) * Y(:,9)) + (RC(:,331)) &
                + (RC(:,139) * Y(:,8)) + (RC(:,140) * Y(:,8)) + (RC(:,222) * Y(:,5)) 
                Y(:,110) = (YP(:,110) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB7            Y(:,111) 
                P(:) = (RC(:,480) * Y(:,3) * Y(:,202)) &
                + (RC(:,400) * Y(:,3) * Y(:,131)) + (RC(:,444) * Y(:,3) * Y(:,155)) &
                + (RC(:,332) * Y(:,110)) + (RC(:,335) * Y(:,112)) &
                + (RC(:,223) * Y(:,110) * Y(:,5)) + (RC(:,226) * Y(:,112) * Y(:,5)) &
                + (RC(:,140) * Y(:,110) * Y(:,8)) + (RC(:,143) * Y(:,112) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,356) * Y(:,3)) + (DJ(:,18)) 
                Y(:,111) = (YP(:,111) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RU10O2           Y(:,112) 
                P(:) = (RC(:,440) * Y(:,3) * Y(:,168)) + (RC(:,479) * Y(:,202)) &
                + (RC(:,360) * Y(:,3) * Y(:,46)) + (RC(:,361) * Y(:,5) * Y(:,46)) 
                L(:) = 0.0 &
                + (RC(:,335)) + (RC(:,478) * Y(:,4)) &
                + (RC(:,273) * Y(:,9)) + (RC(:,333)) + (RC(:,334)) &
                + (RC(:,224) * Y(:,5)) + (RC(:,225) * Y(:,5)) + (RC(:,226) * Y(:,5)) &
                + (RC(:,141) * Y(:,8)) + (RC(:,142) * Y(:,8)) + (RC(:,143) * Y(:,8)) 
                Y(:,112) = (YP(:,112) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NUCARB12         Y(:,113) 
                P(:) = (DJ(:,72) * Y(:,172)) &
                + (RC(:,339) * Y(:,45)) + (RC(:,441) * Y(:,3) * Y(:,172)) &
                + (RC(:,147) * Y(:,45) * Y(:,8)) + (RC(:,230) * Y(:,45) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,376) * Y(:,3)) + (DJ(:,30)) 
                Y(:,113) = (YP(:,113) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRU12O2          Y(:,114) 
                P(:) = (RC(:,376) * Y(:,3) * Y(:,113)) 
                L(:) = 0.0 &
                + (RC(:,340)) &
                + (RC(:,148) * Y(:,8)) + (RC(:,231) * Y(:,5)) + (RC(:,278) * Y(:,9)) 
                Y(:,114) = (YP(:,114) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NOA              Y(:,115) 
                P(:) = (DJ(:,30) * Y(:,113)) + (DJ(:,73) * Y(:,173)) &
                + (RC(:,340) * Y(:,114)) + (RC(:,442) * Y(:,3) * Y(:,173)) &
                + (RC(:,148) * Y(:,114) * Y(:,8)) + (RC(:,231) * Y(:,114) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,377) * Y(:,3)) + (DJ(:,31)) + (DJ(:,32)) 
                Y(:,115) = (YP(:,115) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN25O2          Y(:,116) 
                P(:) = (RC(:,457) * Y(:,3) * Y(:,177)) + (DJ(:,87) * Y(:,176)) &
                + (RC(:,343) * Y(:,50)) + (RC(:,389) * Y(:,3) * Y(:,52)) &
                + (RC(:,152) * Y(:,50) * Y(:,8)) + (RC(:,234) * Y(:,50) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,282) * Y(:,9)) + (RC(:,344)) &
                + (RC(:,153) * Y(:,8)) + (RC(:,186) * Y(:,8)) + (RC(:,235) * Y(:,5)) 
                Y(:,116) = (YP(:,116) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN24O2          Y(:,117) 
                P(:) = (DJ(:,88) * Y(:,177)) &
                + (RC(:,344) * Y(:,116)) + (RC(:,458) * Y(:,3) * Y(:,178)) &
                + (RC(:,153) * Y(:,116) * Y(:,8)) + (RC(:,235) * Y(:,116) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,345)) &
                + (RC(:,154) * Y(:,8)) + (RC(:,236) * Y(:,5)) + (RC(:,283) * Y(:,9)) 
                Y(:,117) = (YP(:,117) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN23O2          Y(:,118) 
                P(:) = (DJ(:,89) * Y(:,178)) &
                + (RC(:,345) * Y(:,117)) + (RC(:,459) * Y(:,3) * Y(:,179)) &
                + (RC(:,154) * Y(:,117) * Y(:,8)) + (RC(:,236) * Y(:,117) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,346)) &
                + (RC(:,155) * Y(:,8)) + (RC(:,237) * Y(:,5)) + (RC(:,284) * Y(:,9)) 
                Y(:,118) = (YP(:,118) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN14O2          Y(:,119) 
                P(:) = (DJ(:,90) * Y(:,179)) &
                + (RC(:,346) * Y(:,118)) + (RC(:,460) * Y(:,3) * Y(:,180)) &
                + (RC(:,155) * Y(:,118) * Y(:,8)) + (RC(:,237) * Y(:,118) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,347)) &
                + (RC(:,156) * Y(:,8)) + (RC(:,238) * Y(:,5)) + (RC(:,285) * Y(:,9)) 
                Y(:,119) = (YP(:,119) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TNCARB10         Y(:,120) 
                P(:) = (RC(:,347) * Y(:,119)) + (DJ(:,91) * Y(:,180)) &
                + (RC(:,156) * Y(:,119) * Y(:,8)) + (RC(:,238) * Y(:,119) * Y(:,5)) 
                L(:) = 0.0 &
                + (RC(:,386) * Y(:,3)) + (RC(:,388) * Y(:,5)) + (DJ(:,40)) 
                Y(:,120) = (YP(:,120) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN10O2          Y(:,121) 
                P(:) = (RC(:,461) * Y(:,3) * Y(:,181)) &
                + (RC(:,386) * Y(:,3) * Y(:,120)) + (RC(:,388) * Y(:,5) * Y(:,120)) 
                L(:) = 0.0 &
                + (RC(:,348)) &
                + (RC(:,157) * Y(:,8)) + (RC(:,239) * Y(:,5)) + (RC(:,286) * Y(:,9)) 
                Y(:,121) = (YP(:,121) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTX22O2          Y(:,122) 
                P(:) = (RC(:,391) * Y(:,3) * Y(:,58)) 
                L(:) = 0.0 &
                + (RC(:,289) * Y(:,9)) + (RC(:,351)) &
                + (RC(:,163) * Y(:,8)) + (RC(:,189) * Y(:,8)) + (RC(:,242) * Y(:,5)) 
                Y(:,122) = (YP(:,122) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3NO3           Y(:,123) 
                P(:) = (RC(:,166) * Y(:,22) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,392) * Y(:,3)) + (DJ(:,41)) 
                Y(:,123) = (YP(:,123) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H5NO3          Y(:,124) 
                P(:) = (RC(:,167) * Y(:,24) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,393) * Y(:,3)) + (DJ(:,42)) 
                Y(:,124) = (YP(:,124) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN10NO3          Y(:,125) 
                P(:) = (RC(:,168) * Y(:,27) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,394) * Y(:,3)) + (DJ(:,43)) 
                Y(:,125) = (YP(:,125) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          IC3H7NO3         Y(:,126) 
                P(:) = (RC(:,169) * Y(:,26) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,395) * Y(:,3)) + (DJ(:,44)) 
                Y(:,126) = (YP(:,126) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN13NO3          Y(:,127) 
                P(:) = (RC(:,170) * Y(:,29) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,396) * Y(:,3)) + (DJ(:,45)) + (DJ(:,46)) 
                Y(:,127) = (YP(:,127) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN16NO3          Y(:,128) 
                P(:) = (RC(:,171) * Y(:,89) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,397) * Y(:,3)) + (DJ(:,47)) 
                Y(:,128) = (YP(:,128) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN19NO3          Y(:,129) 
                P(:) = (RC(:,172) * Y(:,91) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,398) * Y(:,3)) + (DJ(:,48)) 
                Y(:,129) = (YP(:,129) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HOC2H4NO3        Y(:,130) 
                P(:) = (RC(:,173) * Y(:,31) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,399) * Y(:,3)) 
                Y(:,130) = (YP(:,130) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN9NO3           Y(:,131) 
                P(:) = (RC(:,174) * Y(:,33) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,400) * Y(:,3)) 
                Y(:,131) = (YP(:,131) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN12NO3          Y(:,132) 
                P(:) = (RC(:,175) * Y(:,35) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,401) * Y(:,3)) 
                Y(:,132) = (YP(:,132) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN15NO3          Y(:,133) 
                P(:) = (RC(:,176) * Y(:,95) * Y(:,8)) + (RC(:,178) * Y(:,90) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,402) * Y(:,3)) 
                Y(:,133) = (YP(:,133) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN18NO3          Y(:,134) 
                P(:) = (RC(:,177) * Y(:,103) * Y(:,8)) + (RC(:,179) * Y(:,92) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,403) * Y(:,3)) 
                Y(:,134) = (YP(:,134) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RU14NO3          Y(:,135) 
                P(:) = (RC(:,180) * Y(:,44) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,404) * Y(:,3)) 
                Y(:,135) = (YP(:,135) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA13NO3          Y(:,136) 
                P(:) = (RC(:,181) * Y(:,62) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,405) * Y(:,3)) + (DJ(:,49)) 
                Y(:,136) = (YP(:,136) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA16NO3          Y(:,137) 
                P(:) = (RC(:,182) * Y(:,65) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,406) * Y(:,3)) + (DJ(:,50)) 
                Y(:,137) = (YP(:,137) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA19NO3          Y(:,138) 
                P(:) = (RC(:,183) * Y(:,68) * Y(:,8)) + (RC(:,184) * Y(:,69) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,407) * Y(:,3)) + (DJ(:,51)) 
                Y(:,138) = (YP(:,138) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN28NO3         Y(:,139) 
                P(:) = (RC(:,185) * Y(:,48) * Y(:,8)) + (RC(:,486) * Y(:,204)) 
                L(:) = 0.0 &
                + (RC(:,408) * Y(:,3)) + (RC(:,485)) 
                Y(:,139) = (YP(:,139) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN25NO3         Y(:,140) 
                P(:) = (RC(:,186) * Y(:,116) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,409) * Y(:,3)) 
                Y(:,140) = (YP(:,140) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTX28NO3         Y(:,141) 
                P(:) = (RC(:,187) * Y(:,54) * Y(:,8)) + (RC(:,488) * Y(:,205)) 
                L(:) = 0.0 &
                + (RC(:,410) * Y(:,3)) + (RC(:,487)) 
                Y(:,141) = (YP(:,141) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTX24NO3         Y(:,142) 
                P(:) = (RC(:,188) * Y(:,56) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,411) * Y(:,3)) + (DJ(:,52)) 
                Y(:,142) = (YP(:,142) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTX22NO3         Y(:,143) 
                P(:) = (RC(:,189) * Y(:,122) * Y(:,8)) 
                L(:) = 0.0 &
                + (RC(:,412) * Y(:,3)) 
                Y(:,143) = (YP(:,143) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3OOH           Y(:,144) 
                P(:) = (RC(:,244) * Y(:,22) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,423) * Y(:,3)) + (RC(:,424) * Y(:,3)) + (DJ(:,53)) 
                Y(:,144) = (YP(:,144) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H5OOH          Y(:,145) 
                P(:) = (RC(:,245) * Y(:,24) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,425) * Y(:,3)) + (DJ(:,54)) 
                Y(:,145) = (YP(:,145) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN10OOH          Y(:,146) 
                P(:) = (RC(:,246) * Y(:,27) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,426) * Y(:,3)) + (DJ(:,55)) 
                Y(:,146) = (YP(:,146) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          IC3H7OOH         Y(:,147) 
                P(:) = (RC(:,247) * Y(:,26) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,427) * Y(:,3)) + (DJ(:,56)) 
                Y(:,147) = (YP(:,147) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN13OOH          Y(:,148) 
                P(:) = (RC(:,248) * Y(:,29) * Y(:,9)) + (RC(:,251) * Y(:,93) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,428) * Y(:,3)) + (DJ(:,57)) + (DJ(:,58)) 
                Y(:,148) = (YP(:,148) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN16OOH          Y(:,149) 
                P(:) = (RC(:,249) * Y(:,89) * Y(:,9)) + (RC(:,252) * Y(:,94) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,429) * Y(:,3)) + (DJ(:,59)) 
                Y(:,149) = (YP(:,149) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN19OOH          Y(:,150) 
                P(:) = (RC(:,250) * Y(:,91) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,430) * Y(:,3)) + (DJ(:,60)) 
                Y(:,150) = (YP(:,150) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA13OOH          Y(:,151) 
                P(:) = (RC(:,253) * Y(:,62) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,451) * Y(:,3)) + (DJ(:,82)) 
                Y(:,151) = (YP(:,151) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA16OOH          Y(:,152) 
                P(:) = (RC(:,254) * Y(:,65) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,452) * Y(:,3)) + (DJ(:,83)) 
                Y(:,152) = (YP(:,152) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RA19OOH          Y(:,153) 
                P(:) = (RC(:,255) * Y(:,68) * Y(:,9)) + (RC(:,256) * Y(:,69) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,453) * Y(:,3)) + (DJ(:,84)) 
                Y(:,153) = (YP(:,153) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HOC2H4OOH        Y(:,154) 
                P(:) = (RC(:,257) * Y(:,31) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,443) * Y(:,3)) + (DJ(:,74)) 
                Y(:,154) = (YP(:,154) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN9OOH           Y(:,155) 
                P(:) = (RC(:,258) * Y(:,33) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,444) * Y(:,3)) + (DJ(:,75)) 
                Y(:,155) = (YP(:,155) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN12OOH          Y(:,156) 
                P(:) = (RC(:,259) * Y(:,35) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,445) * Y(:,3)) + (DJ(:,76)) 
                Y(:,156) = (YP(:,156) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN15OOH          Y(:,157) 
                P(:) = (RC(:,260) * Y(:,95) * Y(:,9)) + (RC(:,262) * Y(:,90) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,446) * Y(:,3)) + (DJ(:,77)) 
                Y(:,157) = (YP(:,157) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN18OOH          Y(:,158) 
                P(:) = (RC(:,261) * Y(:,103) * Y(:,9)) + (RC(:,263) * Y(:,92) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,447) * Y(:,3)) + (DJ(:,78)) 
                Y(:,158) = (YP(:,158) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3CO3H          Y(:,159) 
                P(:) = (RC(:,264) * Y(:,70) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,431) * Y(:,3)) + (DJ(:,61)) 
                Y(:,159) = (YP(:,159) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          C2H5CO3H         Y(:,160) 
                P(:) = (RC(:,265) * Y(:,72) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,432) * Y(:,3)) + (DJ(:,62)) 
                Y(:,160) = (YP(:,160) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          HOCH2CO3H        Y(:,161) 
                P(:) = (RC(:,266) * Y(:,106) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,433) * Y(:,3)) + (DJ(:,63)) 
                Y(:,161) = (YP(:,161) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN8OOH           Y(:,162) 
                P(:) = (RC(:,267) * Y(:,74) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,434) * Y(:,3)) + (DJ(:,64)) 
                Y(:,162) = (YP(:,162) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN11OOH          Y(:,163) 
                P(:) = (RC(:,268) * Y(:,75) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,435) * Y(:,3)) + (DJ(:,65)) 
                Y(:,163) = (YP(:,163) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN14OOH          Y(:,164) 
                P(:) = (RC(:,269) * Y(:,107) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,436) * Y(:,3)) + (DJ(:,66)) 
                Y(:,164) = (YP(:,164) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RN17OOH          Y(:,165) 
                P(:) = (RC(:,270) * Y(:,108) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,437) * Y(:,3)) + (DJ(:,67)) 
                Y(:,165) = (YP(:,165) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RU14OOH          Y(:,166) 
                P(:) = (RC(:,271) * Y(:,44) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,438) * Y(:,3)) + (DJ(:,68)) + (DJ(:,69)) 
                Y(:,166) = (YP(:,166) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RU12OOH          Y(:,167) 
                P(:) = (RC(:,272) * Y(:,110) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,439) * Y(:,3)) + (DJ(:,70)) 
                Y(:,167) = (YP(:,167) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RU10OOH          Y(:,168) 
                P(:) = (RC(:,273) * Y(:,112) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,440) * Y(:,3)) + (DJ(:,71)) 
                Y(:,168) = (YP(:,168) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRN6OOH          Y(:,169) 
                P(:) = (RC(:,274) * Y(:,36) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,448) * Y(:,3)) + (DJ(:,79)) 
                Y(:,169) = (YP(:,169) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRN9OOH          Y(:,170) 
                P(:) = (RC(:,275) * Y(:,37) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,449) * Y(:,3)) + (DJ(:,80)) 
                Y(:,170) = (YP(:,170) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRN12OOH         Y(:,171) 
                P(:) = (RC(:,276) * Y(:,38) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,450) * Y(:,3)) + (DJ(:,81)) 
                Y(:,171) = (YP(:,171) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRU14OOH         Y(:,172) 
                P(:) = (RC(:,277) * Y(:,45) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,441) * Y(:,3)) + (DJ(:,72)) 
                Y(:,172) = (YP(:,172) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRU12OOH         Y(:,173) 
                P(:) = (RC(:,278) * Y(:,114) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,442) * Y(:,3)) + (DJ(:,73)) 
                Y(:,173) = (YP(:,173) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN28OOH         Y(:,174) 
                P(:) = (RC(:,279) * Y(:,48) * Y(:,9)) + (RC(:,496) * Y(:,209)) 
                L(:) = 0.0 &
                + (RC(:,454) * Y(:,3)) + (RC(:,495)) + (DJ(:,85)) 
                Y(:,174) = (YP(:,174) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRTN28OOH        Y(:,175) 
                P(:) = (RC(:,280) * Y(:,49) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,456) * Y(:,3)) + (DJ(:,86)) 
                Y(:,175) = (YP(:,175) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN26OOH         Y(:,176) 
                P(:) = (RC(:,281) * Y(:,50) * Y(:,9)) + (RC(:,498) * Y(:,210)) 
                L(:) = 0.0 &
                + (RC(:,455) * Y(:,3)) + (RC(:,497)) + (DJ(:,87)) 
                Y(:,176) = (YP(:,176) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN25OOH         Y(:,177) 
                P(:) = (RC(:,282) * Y(:,116) * Y(:,9)) + (RC(:,502) * Y(:,212)) 
                L(:) = 0.0 &
                + (RC(:,457) * Y(:,3)) + (RC(:,501)) + (DJ(:,88)) 
                Y(:,177) = (YP(:,177) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN24OOH         Y(:,178) 
                P(:) = (RC(:,283) * Y(:,117) * Y(:,9)) + (RC(:,492) * Y(:,207)) 
                L(:) = 0.0 &
                + (RC(:,458) * Y(:,3)) + (RC(:,491)) + (DJ(:,89)) 
                Y(:,178) = (YP(:,178) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN23OOH         Y(:,179) 
                P(:) = (RC(:,284) * Y(:,118) * Y(:,9)) + (RC(:,504) * Y(:,213)) 
                L(:) = 0.0 &
                + (RC(:,459) * Y(:,3)) + (RC(:,503)) + (DJ(:,90)) 
                Y(:,179) = (YP(:,179) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN14OOH         Y(:,180) 
                P(:) = (RC(:,285) * Y(:,119) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,460) * Y(:,3)) + (DJ(:,91)) 
                Y(:,180) = (YP(:,180) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN10OOH         Y(:,181) 
                P(:) = (RC(:,286) * Y(:,121) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,461) * Y(:,3)) + (DJ(:,92)) 
                Y(:,181) = (YP(:,181) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTX28OOH         Y(:,182) 
                P(:) = (RC(:,287) * Y(:,54) * Y(:,9)) + (RC(:,494) * Y(:,208)) 
                L(:) = 0.0 &
                + (RC(:,462) * Y(:,3)) + (RC(:,493)) + (DJ(:,93)) 
                Y(:,182) = (YP(:,182) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTX24OOH         Y(:,183) 
                P(:) = (RC(:,288) * Y(:,56) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,463) * Y(:,3)) + (DJ(:,94)) 
                Y(:,183) = (YP(:,183) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTX22OOH         Y(:,184) 
                P(:) = (RC(:,289) * Y(:,122) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,464) * Y(:,3)) + (DJ(:,95)) 
                Y(:,184) = (YP(:,184) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          NRTX28OOH        Y(:,185) 
                P(:) = (RC(:,290) * Y(:,55) * Y(:,9)) 
                L(:) = 0.0 &
                + (RC(:,465) * Y(:,3)) + (DJ(:,96)) 
                Y(:,185) = (YP(:,185) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB14           Y(:,186) 
                P(:) = (RC(:,397) * Y(:,3) * Y(:,128)) + (RC(:,429) * Y(:,3) * Y(:,149)) 
                L(:) = 0.0 &
                + (RC(:,353) * Y(:,3)) + (DJ(:,15)) 
                Y(:,186) = (YP(:,186) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB17           Y(:,187) 
                P(:) = (RC(:,398) * Y(:,3) * Y(:,129)) + (RC(:,430) * Y(:,3) * Y(:,150)) 
                L(:) = 0.0 &
                + (RC(:,354) * Y(:,3)) + (DJ(:,16)) 
                Y(:,187) = (YP(:,187) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB10           Y(:,188) 
                P(:) = (RC(:,401) * Y(:,3) * Y(:,132)) + (RC(:,445) * Y(:,3) * Y(:,156)) 
                L(:) = 0.0 &
                + (RC(:,357) * Y(:,3)) + (DJ(:,19)) 
                Y(:,188) = (YP(:,188) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB12           Y(:,189) 
                P(:) = (RC(:,436) * Y(:,3) * Y(:,164)) 
                L(:) = 0.0 &
                + (RC(:,369) * Y(:,3)) + (DJ(:,27)) 
                Y(:,189) = (YP(:,189) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CARB15           Y(:,190) 
                P(:) = (RC(:,437) * Y(:,3) * Y(:,165)) 
                L(:) = 0.0 &
                + (RC(:,370) * Y(:,3)) + (DJ(:,28)) 
                Y(:,190) = (YP(:,190) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CCARB12          Y(:,191) 
                P(:) = (RC(:,412) * Y(:,3) * Y(:,143)) + (RC(:,464) * Y(:,3) * Y(:,184)) 
                L(:) = 0.0 &
                + (RC(:,371) * Y(:,3)) 
                Y(:,191) = (YP(:,191) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          ANHY             Y(:,192) 
                P(:) = (DJ(:,38) * Y(:,99)) &
                + (DJ(:,34) * Y(:,96)) + (DJ(:,36) * Y(:,97)) &
                + (RC(:,383) * Y(:,3) * Y(:,99)) + (RC(:,510) * Y(:,216)) &
                + (RC(:,379) * Y(:,3) * Y(:,96)) + (RC(:,381) * Y(:,3) * Y(:,97)) 
                L(:) = 0.0 &
                + (RC(:,466) * Y(:,3)) + (RC(:,509)) 
                Y(:,192) = (YP(:,192) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          TNCARB15         Y(:,193) 
                P(:) = (RC(:,409) * Y(:,3) * Y(:,140)) 
                L(:) = 0.0 &
                + (RC(:,385) * Y(:,3)) 
                Y(:,193) = (YP(:,193) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RAROH14          Y(:,194) 
                P(:) = (RC(:,413) * Y(:,3) * Y(:,63)) + (RC(:,414) * Y(:,5) * Y(:,63)) 
                L(:) = 0.0 &
                + (RC(:,415) * Y(:,4)) 
                Y(:,194) = (YP(:,194) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          ARNOH14          Y(:,195) 
                P(:) = (RC(:,415) * Y(:,194) * Y(:,4)) + (RC(:,506) * Y(:,214)) 
                L(:) = 0.0 &
                + (RC(:,416) * Y(:,3)) + (RC(:,417) * Y(:,5)) + (RC(:,505)) 
                Y(:,195) = (YP(:,195) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RAROH17          Y(:,196) 
                P(:) = (RC(:,418) * Y(:,3) * Y(:,66)) + (RC(:,419) * Y(:,5) * Y(:,66)) 
                L(:) = 0.0 &
                + (RC(:,420) * Y(:,4)) 
                Y(:,196) = (YP(:,196) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          ARNOH17          Y(:,197) 
                P(:) = (RC(:,420) * Y(:,196) * Y(:,4)) + (RC(:,508) * Y(:,215)) 
                L(:) = 0.0 &
                + (RC(:,421) * Y(:,3)) + (RC(:,422) * Y(:,5)) + (RC(:,507)) 
                Y(:,197) = (YP(:,197) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          PAN              Y(:,198) 
                P(:) = (RC(:,467) * Y(:,70) * Y(:,4)) 
                L(:) = 0.0 &
                + (RC(:,468)) + (RC(:,473) * Y(:,3)) 
                Y(:,198) = (YP(:,198) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          PPN              Y(:,199) 
                P(:) = (RC(:,469) * Y(:,72) * Y(:,4)) 
                L(:) = 0.0 &
                + (RC(:,470)) + (RC(:,474) * Y(:,3)) 
                Y(:,199) = (YP(:,199) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          PHAN             Y(:,200) 
                P(:) = (RC(:,471) * Y(:,106) * Y(:,4)) 
                L(:) = 0.0 &
                + (RC(:,472)) + (RC(:,475) * Y(:,3)) 
                Y(:,200) = (YP(:,200) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RU12PAN          Y(:,201) 
                P(:) = (RC(:,476) * Y(:,110) * Y(:,4)) 
                L(:) = 0.0 &
                + (RC(:,477)) + (RC(:,481) * Y(:,3)) 
                Y(:,201) = (YP(:,201) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          MPAN             Y(:,202) 
                P(:) = (RC(:,478) * Y(:,112) * Y(:,4)) 
                L(:) = 0.0 &
                + (RC(:,479)) + (RC(:,480) * Y(:,3)) 
                Y(:,202) = (YP(:,202) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          RTN26PAN         Y(:,203) 
                P(:) = (RC(:,482) * Y(:,50) * Y(:,4)) + (RC(:,500) * Y(:,211)) 
                L(:) = 0.0 &
                + (RC(:,483)) + (RC(:,484) * Y(:,3)) + (RC(:,499)) 
                Y(:,203) = (YP(:,203) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P2604            Y(:,204) 
                P(:) = (RC(:,485) * Y(:,139)) 
                L(:) = 0.0 &
                + (RC(:,486)) 
                Y(:,204) = (YP(:,204) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P4608            Y(:,205) 
                P(:) = (RC(:,487) * Y(:,141)) 
                L(:) = 0.0 &
                + (RC(:,488)) 
                Y(:,205) = (YP(:,205) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P2631            Y(:,206) 
                P(:) = (RC(:,489) * Y(:,52)) 
                L(:) = 0.0 &
                + (RC(:,490)) 
                Y(:,206) = (YP(:,206) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P2635            Y(:,207) 
                P(:) = (RC(:,491) * Y(:,178)) 
                L(:) = 0.0 &
                + (RC(:,492)) 
                Y(:,207) = (YP(:,207) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P4610            Y(:,208) 
                P(:) = (RC(:,493) * Y(:,182)) 
                L(:) = 0.0 &
                + (RC(:,494)) 
                Y(:,208) = (YP(:,208) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P2605            Y(:,209) 
                P(:) = (RC(:,495) * Y(:,174)) 
                L(:) = 0.0 &
                + (RC(:,496)) 
                Y(:,209) = (YP(:,209) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P2630            Y(:,210) 
                P(:) = (RC(:,497) * Y(:,176)) 
                L(:) = 0.0 &
                + (RC(:,498)) 
                Y(:,210) = (YP(:,210) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P2629            Y(:,211) 
                P(:) = (RC(:,499) * Y(:,203)) 
                L(:) = 0.0 &
                + (RC(:,500)) 
                Y(:,211) = (YP(:,211) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P2632            Y(:,212) 
                P(:) = (RC(:,501) * Y(:,177)) 
                L(:) = 0.0 &
                + (RC(:,502)) 
                Y(:,212) = (YP(:,212) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P2637            Y(:,213) 
                P(:) = (RC(:,503) * Y(:,179)) 
                L(:) = 0.0 &
                + (RC(:,504)) 
                Y(:,213) = (YP(:,213) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P3612            Y(:,214) 
                P(:) = (RC(:,505) * Y(:,195)) 
                L(:) = 0.0 &
                + (RC(:,506)) 
                Y(:,214) = (YP(:,214) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P3613            Y(:,215) 
                P(:) = (RC(:,507) * Y(:,197)) 
                L(:) = 0.0 &
                + (RC(:,508)) 
                Y(:,215) = (YP(:,215) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P3442            Y(:,216) 
                P(:) = (RC(:,509) * Y(:,192)) 
                L(:) = 0.0 &
                + (RC(:,510)) 
                Y(:,216) = (YP(:,216) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          CH3O2NO2         Y(:,217) 
                P(:) = (RC(:,164) * Y(:,22) * Y(:,4)) 
                L(:) = 0.0 &
                + (RC(:,165)) 
                Y(:,217) = (YP(:,217) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          EMPOA            Y(:,218) 
                P(:) = 0.0 
                !     dry deposition of EMPOA
                L(:) = 0.0
                Y(:,218) = (YP(:,218) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !          P2007            Y(:,219) 
                ! P(:) = EM(:,219) &
                ! + (RC(:,511) * Y(:,167)) 
                ! L(:) = 0.0 &
                ! + (RC(:,512)) 
                ! Y(:,219) = (YP(:,219) + DTS * P(:)) /(1.0+ DTS * L(:)) 

                !  DEFINE TOTAL CONCENTRATION OF PEROXY RADICALS

                RO2(:) =  Y(:,22) + Y(:,24)+ Y(:,27) + Y(:,26) + Y(:,29) + &
                Y(:,93) + Y(:,94) + Y(:,89) + Y(:,91) + Y(:,31) + Y(:,33) + &
                Y(:,35) + Y(:,95) + Y(:,103) + Y(:,90) + Y(:,92) + Y(:,70) + &
                Y(:,72) + Y(:,75) + Y(:,107) + Y(:,108) + Y(:,106) + Y(:,44) + &
                Y(:,110) + Y(:,112) + Y(:,36) + Y(:,37) + Y(:,38) + Y(:,48) + &
                Y(:,45) + Y(:,114) + Y(:,62) + Y(:,65) + Y(:,68) + Y(:,69) + &
                Y(:,74) + Y(:,50) + Y(:,49) + Y(:,116) + Y(:,117) + Y(:,118) + &
                Y(:,119) + Y(:,121) + Y(:,54) + Y(:,56) + Y(:,122) + Y(:,55) 

                ! CALCULATE FLUX TERMS

                !      O + O2 + M = O3 + M
                FL(:,1)=FL(:,1)+RC(:,1)*Y(:,2)*DTS
                !      O + N2 + M = O3 + M
                FL(:,2)=FL(:,2)+RC(:,2)*Y(:,2)*DTS
                !      O + O3 = 
                FL(:,3)=FL(:,3)+RC(:,3)*Y(:,2)*Y(:,6)*DTS
                !       O + NO = NO2
                FL(:,4)=FL(:,4)+RC(:,4)*Y(:,2)*Y(:,8)*DTS
                !      O + NO2 = NO
                FL(:,5)=FL(:,5)+RC(:,5)*Y(:,2)*Y(:,4)*DTS
                !      O + NO2 = NO3
                FL(:,6)=FL(:,6)+RC(:,6)*Y(:,2)*Y(:,4)*DTS
                !      O1D + O2 + M = O + M
                FL(:,7)=FL(:,7)+RC(:,7)*Y(:,1)*DTS
                !      O1D + N2 + M = O + M
                FL(:,8)=FL(:,8)+RC(:,8)*Y(:,1)*DTS
                !      NO + O3 = NO2
                FL(:,9)=FL(:,9) + RC(:,9)*Y(:,8)*Y(:,6)*DTS
                !      NO2 + O3 = NO3
                FL(:,10)=FL(:,10)+RC(:,10)*Y(:,4)*Y(:,6)*DTS
                !      NO + NO = NO2 + NO2
                FL(:,11)=FL(:,11)+RC(:,11)*Y(:,4)*Y(:,4)*DTS
                !      NO + NO3 = NO2 + NO2
                FL(:,12)=FL(:,12)+RC(:,12)*Y(:,4)*Y(:,5)*DTS
                !      NO2 + NO3 = NO + NO2
                FL(:,13)=FL(:,13)+RC(:,13)*Y(:,4)*Y(:,5)*DTS
                !      NO2 + NO3 = N2O5
                FL(:,14)=FL(:,14)+RC(:,14)*Y(:,4)*Y(:,5)*DTS
                !      N2O5 = NO2 + NO3
                FL(:,15)=FL(:,15)+RC(:,15)*Y(:,7)*DTS
                !      O1D = OH + OH
                FL(:,16)=FL(:,16)+RC(:,16)*Y(:,1)*DTS
                !      OH + O3 = HO2
                FL(:,17)=FL(:,17)+RC(:,17)*Y(:,3)*Y(:,6)*DTS
                !      OH + H2 = HO2
                FL(:,18)=FL(:,18)+RC(:,18)*Y(:,3)*Y(:,10)*DTS
                !       OH + CO = HO2
                FL(:,19)=FL(:,19)+RC(:,19)*Y(:,3)*Y(:,11)*DTS
                !      OH + H2O2 = HO2
                FL(:,20)=FL(:,20)+RC(:,20)*Y(:,3)*Y(:,12)*DTS
                !      HO2 + O3 = OH
                FL(:,21)=FL(:,21)+RC(:,21)*Y(:,9)*Y(:,6)*DTS
                !      OH + HO2 = 
                FL(:,22)=FL(:,22)+RC(:,22)*Y(:,3)*Y(:,9)*DTS
                !      HO2 + HO2 = H2O2
                FL(:,23)=FL(:,23)+RC(:,23)*Y(:,9)*Y(:,9)*DTS
                !      HO2 + HO2 = H2O2
                FL(:,24)=FL(:,24)+RC(:,24)*Y(:,9)*Y(:,9)*DTS
                !      OH + NO = HONO
                FL(:,25)=FL(:,25)+RC(:,25)*Y(:,3)*Y(:,8)*DTS
                !      NO2 = HONO
                FL(:,26)=FL(:,26)+RC(:,26)*Y(:,4)*DTS
                !      OH + NO2 = HNO3
                FL(:,27)=FL(:,27)+RC(:,27)*Y(:,3)*Y(:,4)*DTS
                !      OH + NO3 = HO2 + NO2
                FL(:,28)=FL(:,28)+RC(:,28)*Y(:,3)*Y(:,5)*DTS
                !      HO2 + NO = OH + NO2
                FL(:,29)=FL(:,29)+RC(:,29)*Y(:,9)*Y(:,8)*DTS
                !      HO2 + NO2 = HO2NO2
                FL(:,30)=FL(:,30)+RC(:,30)*Y(:,9)*Y(:,4)*DTS
                !      HO2NO2 = HO2 + NO2
                FL(:,31)=FL(:,31)+RC(:,31)*Y(:,15)*DTS
                !      OH + HO2NO2 = NO2 
                FL(:,32)=FL(:,32)+RC(:,32)*Y(:,3)*Y(:,15)*DTS
                !      HO2 + NO3 = OH + NO2
                FL(:,33)=FL(:,33)+RC(:,33)*Y(:,9)*Y(:,5)*DTS
                !      OH + HONO = NO2
                FL(:,34)=FL(:,34)+RC(:,34)*Y(:,3)*Y(:,13)*DTS
                !      OH + HNO3 = NO3
                FL(:,35)=FL(:,35)+RC(:,35)*Y(:,3)*Y(:,14)*DTS
                !      O + SO2 = SO3
                FL(:,36)=FL(:,36)+RC(:,36)*Y(:,2)*Y(:,16)*DTS
                !      OH + SO2 = HSO3 
                FL(:,37)=FL(:,37)+RC(:,37)*Y(:,3)*Y(:,16)*DTS
                !      HSO3 = HO2 + SO3
                FL(:,38)=FL(:,38)+RC(:,38)*Y(:,18)*DTS
                !      HNO3 = NA
                FL(:,39)=FL(:,39)+RC(:,39)*Y(:,14)*DTS
                !      N2O5 = NA + NA
                FL(:,40)=FL(:,40)+RC(:,40)*Y(:,7)*DTS
                !      SO3 = SA
                FL(:,41)=FL(:,41)+RC(:,41)*Y(:,17)*DTS
                !      OH + CH4 = CH3O2
                FL(:,42)=FL(:,42)+RC(:,42)*Y(:,3)*Y(:,21)*DTS
                !      OH + C2H6 = C2H5O2
                FL(:,43)=FL(:,43)+RC(:,43)*Y(:,3)*Y(:,26)*DTS
                !      OH + C3H8 = IC3H7O2
                FL(:,44)=FL(:,44)+RC(:,44)*Y(:,3)*Y(:,25)*DTS
                !      OH + C3H8 = RN10O2 
                FL(:,45)=FL(:,45)+RC(:,45)*Y(:,3)*Y(:,25)*DTS
                !      OH + NC4H10 = RN13O2
                FL(:,46)=FL(:,46)+RC(:,46)*Y(:,3)*Y(:,28)*DTS
                !      OH + C2H4 = HOCH2CH2O2
                FL(:,47)=FL(:,47)+RC(:,47)*Y(:,3)*Y(:,30)*DTS
                !      OH + C3H6 = RN9O2
                FL(:,48)=FL(:,48)+RC(:,48)*Y(:,3)*Y(:,32)*DTS
                !      OH + TBUT2ENE = RN12O2
                FL(:,49)=FL(:,49)+RC(:,49)*Y(:,3)*Y(:,34)*DTS
                !      NO3 + C2H4 = NRN6O2
                FL(:,50)=FL(:,50)+RC(:,50)*Y(:,3)*Y(:,30)*DTS
                !      NO3 + C3H6 = NRN9O2
                FL(:,51)=FL(:,51)+RC(:,51)*Y(:,5)*Y(:,32)*DTS
                !      NO3 + TBUT2ENE = NRN12O2
                FL(:,52)=FL(:,52)+RC(:,52)*Y(:,5)*Y(:,34)*DTS
                !      O3 + C2H4 = HCHO + CO + HO2 + OH
                FL(:,53)=FL(:,53)+RC(:,53)*Y(:,6)*Y(:,30)*DTS
                !      O3 + C2H4 = HCHO + HCOOH
                FL(:,54)=FL(:,54)+RC(:,54)*Y(:,6)*Y(:,30)*DTS
                !      O3 + C3H6 = HCHO + CO + CH3O2 + OH
                FL(:,55)=FL(:,55)+RC(:,55)*Y(:,6)*Y(:,32)*DTS
                !      O3 + C3H6 = HCHO + CH3CO2H
                FL(:,56)=FL(:,56)+RC(:,56)*Y(:,6)*Y(:,32)*DTS
                !      O3 + TBUT2ENE = CH3CHO + CO + CH3O2 + OH
                FL(:,57)=FL(:,57)+RC(:,57)*Y(:,6)*Y(:,34)*DTS
                !      O3 + TBUT2ENE = CH3CHO + CH3CO2H
                FL(:,58)=FL(:,58)+RC(:,58)*Y(:,6)*Y(:,34)*DTS
                !      OH + C5H8 = RU14O2
                FL(:,59)=FL(:,59)+RC(:,59)*Y(:,3)*Y(:,43)*DTS
                !      NO3 + C5H8 = NRU14O2
                FL(:,60)=FL(:,60)+RC(:,60)*Y(:,5)*Y(:,43)*DTS
                !      O3 + C5H8 = UCARB10 + CO + HO2 + OH
                FL(:,61)=FL(:,61)+RC(:,61)*Y(:,6)*Y(:,43)*DTS
                !      O3 + C5H8 = UCARB10 + HCOOH
                FL(:,62)=FL(:,62)+RC(:,62)*Y(:,6)*Y(:,43)*DTS
                !      APINENE + OH = RTN28O2
                FL(:,63)=FL(:,63)+RC(:,63)*Y(:,47)*Y(:,3)*DTS
                !      APINENE + NO3 = NRTN28O2
                FL(:,64)=FL(:,64)+RC(:,64)*Y(:,47)*Y(:,5)*DTS
                !      APINENE + O3 = OH + RTN26O2 
                FL(:,65)=FL(:,65)+RC(:,65)*Y(:,47)*Y(:,6)*DTS
                !      APINENE + O3 = TNCARB26 + H2O2
                FL(:,66)=FL(:,66)+RC(:,66)*Y(:,47)*Y(:,6)*DTS
                !      APINENE + O3 = RCOOH25 
                FL(:,67)=FL(:,67)+RC(:,67)*Y(:,47)*Y(:,6)*DTS
                !      BPINENE + OH = RTX28O2
                FL(:,68)=FL(:,68)+RC(:,68)*Y(:,53)*Y(:,3)*DTS
                !      BPINENE + NO3 = NRTX28O2
                FL(:,69)=FL(:,69)+RC(:,69)*Y(:,53)*Y(:,5)*DTS
                !      BPINENE + O3 =  RTX24O2 + OH
                FL(:,70)=FL(:,70)+RC(:,70)*Y(:,53)*Y(:,6)*DTS
                !      BPINENE + O3 =  HCHO + TXCARB24 + H2O2
                FL(:,71)=FL(:,71)+RC(:,71)*Y(:,53)*Y(:,6)*DTS
                !      BPINENE + O3 =  HCHO + TXCARB22
                FL(:,72)=FL(:,72)+RC(:,72)*Y(:,53)*Y(:,6)*DTS
                !      BPINENE + O3 =  TXCARB24 + CO 
                FL(:,73)=FL(:,73)+RC(:,73)*Y(:,53)*Y(:,6)*DTS
                !      C2H2 + OH = HCOOH + CO + HO2
                FL(:,74)=FL(:,74)+RC(:,74)*Y(:,59)*Y(:,3)*DTS
                !      C2H2 + OH = CARB3 + OH
                FL(:,75)=FL(:,75)+RC(:,75)*Y(:,59)*Y(:,3)*DTS
                !      BENZENE + OH = RA13O2
                FL(:,76)=FL(:,76)+RC(:,76)*Y(:,61)*Y(:,3)*DTS
                !      BENZENE + OH = AROH14 + HO2
                FL(:,77)=FL(:,77)+RC(:,77)*Y(:,61)*Y(:,3)*DTS
                !      TOLUENE + OH = RA16O2
                FL(:,78)=FL(:,78)+RC(:,78)*Y(:,64)*Y(:,3)*DTS
                !      TOLUENE + OH = AROH17 + HO2
                FL(:,79)=FL(:,79)+RC(:,79)*Y(:,64)*Y(:,3)*DTS
                !      OXYL + OH = RA19AO2
                FL(:,80)=FL(:,80)+RC(:,80)*Y(:,67)*Y(:,3)*DTS
                !      OXYL + OH = RA19CO2
                FL(:,81)=FL(:,81)+RC(:,81)*Y(:,67)*Y(:,3)*DTS
                !      OH + HCHO = HO2 + CO
                FL(:,82)=FL(:,82)+RC(:,82)*Y(:,3)*Y(:,39)*DTS
                !      OH + CH3CHO = CH3CO3
                FL(:,83)=FL(:,83)+RC(:,83)*Y(:,3)*Y(:,42)*DTS
                !      OH + C2H5CHO = C2H5CO3
                FL(:,84)=FL(:,84)+RC(:,84)*Y(:,3)*Y(:,71)*DTS
                !      NO3 + HCHO = HO2 + CO + HNO3
                FL(:,85)=FL(:,85)+RC(:,85)*Y(:,5)*Y(:,39)*DTS
                !      NO3 + CH3CHO = CH3CO3 + HNO3
                FL(:,86)=FL(:,86)+RC(:,86)*Y(:,5)*Y(:,42)*DTS
                !      NO3 + C2H5CHO = C2H5CO3 + HNO3
                FL(:,87)=FL(:,87)+RC(:,87)*Y(:,5)*Y(:,71)*DTS
                !      OH + CH3COCH3 = RN8O2
                FL(:,88)=FL(:,88)+RC(:,88)*Y(:,3)*Y(:,73)*DTS
                !      MEK + OH = RN11O2
                FL(:,89)=FL(:,89)+RC(:,89)*Y(:,101)*Y(:,3)*DTS
                !      OH + CH3OH = HO2 + HCHO
                FL(:,90)=FL(:,90)+RC(:,90)*Y(:,3)*Y(:,76)*DTS
                !      OH + C2H5OH = CH3CHO + HO2
                FL(:,91)=FL(:,91)+RC(:,91)*Y(:,3)*Y(:,76)*DTS
                !      OH + C2H5OH = HOCH2CH2O2 
                FL(:,92)=FL(:,92)+RC(:,92)*Y(:,3)*Y(:,77)*DTS
                !     NPROPOL + OH = C2H5CHO + HO2 
                FL(:,93)=FL(:,93)+RC(:,93)*Y(:,3)*Y(:,78)*DTS
                !      NPROPOL + OH = RN9O2
                FL(:,94)=FL(:,94)+RC(:,94)*Y(:,3)*Y(:,78)*DTS
                !      OH + IPROPOL = CH3COCH3 + HO2
                FL(:,95)=FL(:,95)+RC(:,95)*Y(:,3)*Y(:,79)*DTS
                !      OH + IPROPOL = RN9O2
                FL(:,96)=FL(:,96)+RC(:,96)*Y(:,3)*Y(:,79)*DTS
                !      HCOOH + OH = HO2
                FL(:,97)=FL(:,97)+RC(:,97)*Y(:,3)*Y(:,40)*DTS
                !      CH3CO2H + OH = CH3O2
                FL(:,98)=FL(:,98)+RC(:,98)*Y(:,3)*Y(:,41)*DTS
                !      OH + CH3CL = CH3O2 
                FL(:,99)=FL(:,99)+RC(:,99)*Y(:,3)*Y(:,80)*DTS
                !      OH + CH2CL2 = CH3O2
                FL(:,100)=FL(:,100)+RC(:,100)*Y(:,3)*Y(:,81)*DTS
                !      OH + CHCL3 = CH3O2
                FL(:,101)=FL(:,101)+RC(:,101)*Y(:,3)*Y(:,80)*DTS
                !      OH + CH3CCL3 = C2H5O2
                FL(:,102)=FL(:,102)+RC(:,102)*Y(:,3)*Y(:,83)*DTS
                !      OH + TCE = HOCH2CH2O2 
                FL(:,103)=FL(:,103)+RC(:,103)*Y(:,3)*Y(:,84)*DTS
                !      OH + TRICLETH = HOCH2CH2O2
                FL(:,104)=FL(:,104)+RC(:,104)*Y(:,3)*Y(:,85)*DTS
                !      OH + CDICLETH = HOCH2CH2O2
                FL(:,105)=FL(:,105)+RC(:,105)*Y(:,3)*Y(:,86)*DTS
                !      OH + TDICLETH = HOCH2CH2O2
                FL(:,106)=FL(:,106)+RC(:,106)*Y(:,3)*Y(:,87)*DTS
                !      CH3O2 + NO = HCHO + HO2 + NO2
                FL(:,107)=FL(:,107)+RC(:,107)*Y(:,8)*Y(:,22)*DTS
                !      C2H5O2 + NO = CH3CHO + HO2 + NO2
                FL(:,108)=FL(:,108)+RC(:,108)*Y(:,8)*Y(:,24)*DTS
                !      RN10O2 + NO = C2H5CHO + HO2 + NO2
                FL(:,109)=FL(:,109)+RC(:,109)*Y(:,8)*Y(:,27)*DTS
                !      IC3H7O2 + NO = CH3COCH3 + HO2 + NO2
                FL(:,110)=FL(:,110)+RC(:,110)*Y(:,8)*Y(:,26)*DTS
                !      RN13O2 + NO = CH3CHO + C2H5O2 + NO2 
                FL(:,111)=FL(:,111)+RC(:,111)*Y(:,8)*Y(:,29)*DTS
                !      RN13O2 + NO = CARB11A + HO2 + NO2
                FL(:,112)=FL(:,112)+RC(:,112)*Y(:,8)*Y(:,29)*DTS
                !      RN16O2 + NO = RN15AO2 + NO2 
                FL(:,113)=FL(:,113)+RC(:,113)*Y(:,8)*Y(:,89)*DTS
                !      RN19O2 + NO = RN18AO2 + NO2
                FL(:,114)=FL(:,114)+RC(:,114)*Y(:,8)*Y(:,91)*DTS
                !      RN13AO2 + NO = RN12O2 + NO2
                FL(:,115)=FL(:,115)+RC(:,115)*Y(:,8)*Y(:,93)*DTS
                !      RN16AO2 + NO = RN15O2 + NO2 
                FL(:,116)=FL(:,116)+RC(:,116)*Y(:,8)*Y(:,94)*DTS
                !      RA13O2 + NO = CARB3 + UDCARB8 + HO2 + NO2
                FL(:,117)=FL(:,117)+RC(:,117)*Y(:,8)*Y(:,62)*DTS
                !      RA16O2 + NO = CARB3 + UDCARB11 + HO2 + NO2 
                FL(:,118)=FL(:,118)+RC(:,118)*Y(:,8)*Y(:,65)*DTS
                !      RA16O2 + NO = CARB6 + UDCARB8 + HO2 + NO2  
                FL(:,119)=FL(:,119)+RC(:,119)*Y(:,8)*Y(:,65)*DTS
                !      RA19AO2 + NO = CARB3 + UDCARB14 + HO2 + NO2
                FL(:,120)=FL(:,120)+RC(:,120)*Y(:,8)*Y(:,68)*DTS
                !      RA19CO2 + NO = CARB9 + UDCARB8 + HO2 + NO2
                FL(:,121)=FL(:,121)+RC(:,121)*Y(:,8)*Y(:,69)*DTS
                !      HOCH2CH2O2 + NO = HCHO + HCHO + HO2 + NO2
                FL(:,122)=FL(:,122)+RC(:,122)*Y(:,8)*Y(:,31)*DTS
                !      HOCH2CH2O2 + NO = HOCH2CHO + HO2 + NO2
                FL(:,123)=FL(:,123)+RC(:,123)*Y(:,8)*Y(:,31)*DTS
                !       RN9O2 + NO = CH3CHO + HCHO + HO2 + NO2 
                FL(:,124)=FL(:,124)+RC(:,124)*Y(:,8)*Y(:,33)*DTS
                !      RN12O2 + NO = CH3CHO + CH3CHO + HO2 + NO2
                FL(:,125)=FL(:,125)+RC(:,125)*Y(:,8)*Y(:,35)*DTS
                !      RN15O2 + NO = C2H5CHO + CH3CHO + HO2 + NO2
                FL(:,126)=FL(:,126)+RC(:,126)*Y(:,8)*Y(:,95)*DTS
                !      RN18O2 + NO = C2H5CHO + C2H5CHO + HO2 + NO2 
                FL(:,127)=FL(:,127)+RC(:,127)*Y(:,8)*Y(:,103)*DTS
                !      RN15AO2 + NO = CARB13 + HO2 + NO2 
                FL(:,128)=FL(:,128)+RC(:,128)*Y(:,8)*Y(:,90)*DTS
                !      RN18AO2 + NO = CARB16 + HO2 + NO2
                FL(:,129)=FL(:,129)+RC(:,129)*Y(:,8)*Y(:,92)*DTS
                !      CH3CO3 + NO = CH3O2 + NO2 
                FL(:,130)=FL(:,130)+RC(:,130)*Y(:,8)*Y(:,70)*DTS


    END SUBROUTINE DERIV

    SUBROUTINE RUN_LEGACY_CHEM(Y, TEMP, H2O, O2, N2, M, SZA, DTS, NCELL, NSBOXM, NPP, NPC, NTC, NFL)
        IMPLICIT NONE
        
        ! Inputs
        DOUBLE PRECISION, INTENT(INOUT) :: Y(:,:)
        DOUBLE PRECISION, INTENT(IN) :: TEMP(:), H2O(:), O2(:), N2(:), M(:), SZA(:)
        INTEGER, INTENT(IN) :: DTS, NCELL, NSBOXM, NPP, NPC, NTC, NFL
        
        CALL CHEM_ALLOC(NCELL, NSBOXM, NPP, NPC, NTC, NFL)
        ! Step 1: Compute aerosol properties (SOA, MOM)
        CALL CALC_AEROSOL(Y, SOA, MOM)
        ! Step 2: Compute rate coefficients
        CALL CHEMCO(TEMP, N2, O2, M, RC)
        ! Step 3: Compute photolysis rates (J) - takes REAL SZA, outputs REAL J
        CALL CALC_J(NCELL, SZA, J)
        ! Step 4: Compute photolysis derivatives (DJ) from J
        CALL PHOTOL(NCELL, J, DJ)
        ! Step 5: Integrate chemistry (compute Y changes and accumulate FL)
        CALL DERIV(Y, DJ, RC, FL, DTS)

    END SUBROUTINE RUN_LEGACY_CHEM

END MODULE RUN_CHEM_UTILS

MODULE DEFINE_INPUT_TYPES
    USE NETCDF
    USE HELPERS
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: FL_DS_TYPE, PL_DS_TYPE, BOXM_DS_TYPE

    ! ------- INPUT DATA TYPES -------

    TYPE :: FL_DS_TYPE
        ! ---------- PUBLIC DATA YOU ACTUALLY USE ----------

        ! FL DIMLENS
        INTEGER :: NSEG = 0

        ! FL DIMS
        ! INTEGER, ALLOCATABLE :: SEG_ID(:)

        ! FL COORDS
        INTEGER, ALLOCATABLE :: FL_ID(:)
        INTEGER, ALLOCATABLE :: WP(:) 

        ! FL VARS
        REAL(DP), ALLOCATABLE :: LATITUDE(:)
        REAL(DP), ALLOCATABLE :: LONGITUDE(:)
        REAL(DP), ALLOCATABLE :: LEVEL(:)
        REAL(DP), ALLOCATABLE :: ALTITUDE(:)
        
        INTEGER, ALLOCATABLE :: TIME_REL_S(:)
        INTEGER, ALLOCATABLE :: TIME_IDX(:)

        REAL(DP), ALLOCATABLE :: TRUE_AIRSPEED(:)
        REAL(DP), ALLOCATABLE :: AIRCRAFT_MASS(:)
        REAL(DP), ALLOCATABLE :: ENGINE_EFFICIENCY(:)
        REAL(DP), ALLOCATABLE :: FUEL_FLOW(:)
        REAL(DP), ALLOCATABLE :: FUEL_BURN(:)
        REAL(DP), ALLOCATABLE :: THRUST(:)
        REAL(DP), ALLOCATABLE :: ROCD(:)

        ! ---------- PRIVATE NETCDF PLUMBING ----------

        INTEGER, PRIVATE :: FL_NCID = -1

        ! FL DIM IDs
        INTEGER, PRIVATE :: DIMID_SEG_ID

        ! FL VAR IDs
        INTEGER, PRIVATE :: VARID_FL_ID = -1
        INTEGER, PRIVATE :: VARID_WP = -1

        INTEGER, PRIVATE :: VARID_TIME_REL_S = -1
        INTEGER, PRIVATE :: VARID_TIME_IDX = -1
        INTEGER, PRIVATE :: VARID_LATITUDE = -1
        INTEGER, PRIVATE :: VARID_LONGITUDE = -1
        INTEGER, PRIVATE :: VARID_LEVEL = -1
        INTEGER, PRIVATE :: VARID_ALTITUDE = -1
        INTEGER, PRIVATE :: VARID_TRUE_AIRSPEED = -1
        INTEGER, PRIVATE :: VARID_AIRCRAFT_MASS = -1
        INTEGER, PRIVATE :: VARID_ENGINE_EFFICIENCY = -1
        INTEGER, PRIVATE :: VARID_FUEL_FLOW = -1
        INTEGER, PRIVATE :: VARID_FUEL_BURN = -1
        INTEGER, PRIVATE :: VARID_THRUST = -1
        INTEGER, PRIVATE :: VARID_ROCD = -1

        ! NETCDF STATUS
        LOGICAL, PRIVATE :: IS_OPEN = .FALSE.

    CONTAINS
        PROCEDURE, PASS :: INIT        => FL_INIT
        PROCEDURE, PASS :: READ_STATIC => FL_READ_STATIC
        PROCEDURE, PASS :: SUMMARY     => FL_SUMMARY
        PROCEDURE, PASS :: CLOSE       => FL_CLOSE
        FINAL           :: FL_FINALIZE
    
    END TYPE FL_DS_TYPE

    TYPE :: PL_DS_TYPE
        ! ---------- PUBLIC DATA YOU ACTUALLY USE ----------

        ! PL DIMLENS
        INTEGER :: NSEG = 0
        INTEGER :: NSEMI = 0
        INTEGER :: NTPL = 0
      
        ! PL COORDS
        INTEGER, ALLOCATABLE :: FL_ID(:) ! (NSEG)
        INTEGER,   ALLOCATABLE :: WP(:) ! (NSEG)
        INTEGER, ALLOCATABLE :: TIME_REL_S(:) ! (NTPL)
        INTEGER, ALLOCATABLE :: TIME_IDX(:) ! (NTPL)
        INTEGER, ALLOCATABLE :: SPECIES_EMI_NUM(:) ! (NSEMI)
        
        ! PL VARS
        INTEGER, ALLOCATABLE :: ACTIVE_SEG_FLAG(:,:) ! (NSEG, NTPL)
        INTEGER, ALLOCATABLE :: AGE_S(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: LONGITUDE(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: LONGITUDE_M(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: LATITUDE(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: LATITUDE_M(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: LEVEL(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: ALTITUDE(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: WIDTH(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: DEPTH(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: HEADING(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: SIGMA_YY(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: SIGMA_YZ(:,:) ! (NSEG, NTPL)
        REAL(DP), ALLOCATABLE :: SIGMA_ZZ(:,:) ! (NSEG, NTPL)

        REAL(DP), ALLOCATABLE :: EMI_PL_MASS(:,:) ! (NSEG, NSEMI)

        ! ATTRIBUTES
        INTEGER :: NSLICES ! NUMBER OF PLUME SLICES
        REAL(DP) :: FMAX ! MAX FRACTIONAL MASS PER SLICE
        INTEGER :: OUTPUT_PL_SLICES ! WHETHER TO OUTPUT PLUME SLICES TO NETCDF
        INTEGER :: NPOINTS ! NUMBER OF POINTS PER PLUME WAYPOINT ELLIPSE
        INTEGER :: MAX_AGE_S
        
        ! ---------- PRIVATE NETCDF PLUMBING ----------

        ! PL NC IDs
        INTEGER, PRIVATE :: PL_NCID = -1

        ! PL DIM IDs
        INTEGER, PRIVATE :: DIMID_SEG_ID = -1
        INTEGER, PRIVATE :: DIMID_SPECIES_EMI = -1
        INTEGER, PRIVATE :: DIMID_TIME = -1
        
        ! PL VAR IDs
        INTEGER, PRIVATE :: VARID_FL_ID = -1
        INTEGER, PRIVATE :: VARID_WP = -1
        INTEGER, PRIVATE :: VARID_SPECIES_EMI_NUM = -1
        INTEGER, PRIVATE :: VARID_TIME_REL_S = -1
        INTEGER, PRIVATE :: VARID_TIME_IDX = -1

        INTEGER, PRIVATE :: VARID_ACTIVE_SEG_FLAG = -1
        INTEGER, PRIVATE :: VARID_AGE_S = -1
        INTEGER, PRIVATE :: VARID_LONGITUDE = -1
        INTEGER, PRIVATE :: VARID_LONGITUDE_M = -1
        INTEGER, PRIVATE :: VARID_LATITUDE = -1
        INTEGER, PRIVATE :: VARID_LATITUDE_M = -1
        INTEGER, PRIVATE :: VARID_LEVEL = -1
        INTEGER, PRIVATE :: VARID_ALTITUDE = -1
        INTEGER, PRIVATE :: VARID_WIDTH = -1
        INTEGER, PRIVATE :: VARID_DEPTH = -1
        INTEGER, PRIVATE :: VARID_HEADING = -1
        INTEGER, PRIVATE :: VARID_SIGMA_YY = -1
        INTEGER, PRIVATE :: VARID_SIGMA_YZ = -1
        INTEGER, PRIVATE :: VARID_SIGMA_ZZ = -1
        
        INTEGER, PRIVATE :: VARID_EMI_PL_MASS = -1

        ! NETCDF STATUS
        LOGICAL :: IS_OPEN = .FALSE.

    CONTAINS
        PROCEDURE, PASS :: INIT        => PL_INIT
        PROCEDURE, PASS :: READ_STATIC => PL_READ_STATIC
        PROCEDURE, PASS :: SUMMARY     => PL_SUMMARY
        PROCEDURE, PASS :: CLOSE       => PL_CLOSE
        FINAL           :: PL_FINALIZE

    END TYPE PL_DS_TYPE

    TYPE :: BOXM_DS_TYPE
        ! ---------- PUBLIC DATA YOU ACTUALLY USE ----------

        ! BOXM DIMLENS
        INTEGER :: NCELL = 0
        INTEGER :: NSBOXM = 219
        INTEGER :: NTBOXM = 0
        
        ! BOXM COORDS
        INTEGER,   ALLOCATABLE :: TIME_REL_S(:)
        INTEGER,   ALLOCATABLE :: TIME_IDX(:)
        INTEGER,   ALLOCATABLE :: SPECIES_BOXM_NUM(:)
        REAL(DP), ALLOCATABLE :: MOL_MASS_C(:)   ! (NSBOXM) kg/mol

        REAL(DP), ALLOCATABLE :: LONGITUDE_C(:)
        REAL(DP), ALLOCATABLE :: LATITUDE_C(:)
        REAL(DP), ALLOCATABLE :: ALTITUDE_C(:)
        REAL(DP), ALLOCATABLE :: LEVEL_C(:)

        REAL(DP), ALLOCATABLE :: LONGITUDE_C_M(:)
        REAL(DP), ALLOCATABLE :: LATITUDE_C_M(:)
        

        ! BOXM VARS
        REAL(DP), ALLOCATABLE :: TEMP(:,:)
        REAL(DP), ALLOCATABLE :: H2O(:,:)
        REAL(DP), ALLOCATABLE :: M(:,:)
        REAL(DP), ALLOCATABLE :: O2(:,:)
        REAL(DP), ALLOCATABLE :: N2(:,:)
        REAL(DP), ALLOCATABLE :: SZA(:,:)
        REAL(DP), ALLOCATABLE :: Y_BG_C(:,:)

        ! ATTRIBUTES
        INTEGER :: TS_FL
        INTEGER :: TS_PL
        INTEGER :: TS_SIM
        INTEGER :: TS_OUT

        REAL(DP) :: HRES_SIM_C
        REAL(DP) :: VRES_SIM_C
        REAL(DP) :: HRES_SIM_F
        REAL(DP) :: VRES_SIM_F
    
        INTEGER :: NPP ! PHOTOL PARAMS
        INTEGER :: NPC ! PHOTOL COEFFS
        INTEGER :: NTC ! THERMAL COEFFS
        INTEGER :: NFL ! FLUXES

        INTEGER :: RUN_CHEM
        INTEGER :: N_AC

        ! ---------- PRIVATE NETCDF PLUMBING ----------

        ! BOXM NC IDs
        INTEGER, PRIVATE :: BOXM_NCID = -1

        ! BOXM DIM IDs
        INTEGER, PRIVATE :: DIMID_CELL = -1
        INTEGER, PRIVATE :: DIMID_SPECIES_BOXM = -1
        INTEGER, PRIVATE :: DIMID_TIME = -1
        
        ! BOXM VAR IDs
        INTEGER, PRIVATE :: VARID_TIME_REL_S = -1
        INTEGER, PRIVATE :: VARID_TIME_IDX = -1
        INTEGER, PRIVATE :: VARID_SPECIES_BOXM_NUM = -1
        INTEGER, PRIVATE :: VARID_MOL_MASS_C = -1

        INTEGER, PRIVATE :: VARID_LONGITUDE_C = -1
        INTEGER, PRIVATE :: VARID_LATITUDE_C = -1
        INTEGER, PRIVATE :: VARID_ALTITUDE_C = -1
        INTEGER, PRIVATE :: VARID_LEVEL_C = -1

        INTEGER, PRIVATE :: VARID_LONGITUDE_C_M = -1
        INTEGER, PRIVATE :: VARID_LATITUDE_C_M = -1

        INTEGER, PRIVATE :: VARID_TEMP = -1
        INTEGER, PRIVATE :: VARID_H2O = -1
        INTEGER, PRIVATE :: VARID_M = -1
        INTEGER, PRIVATE :: VARID_O2 = -1
        INTEGER, PRIVATE :: VARID_N2 = -1
        INTEGER, PRIVATE :: VARID_SZA = -1
        INTEGER, PRIVATE :: VARID_Y_BG_C = -1

        LOGICAL, PRIVATE :: IS_OPEN = .FALSE.

    CONTAINS
        PROCEDURE, PASS :: INIT        => BOXM_INIT
        PROCEDURE, PASS :: READ_STATIC => BOXM_READ_STATIC
        PROCEDURE, PASS :: SUMMARY     => BOXM_SUMMARY
        PROCEDURE, PASS :: CLOSE       => BOXM_CLOSE
        FINAL           :: BOXM_FINALIZE

    END TYPE BOXM_DS_TYPE

CONTAINS

    ! ---------- FL DS METHODS ----------

    SUBROUTINE FL_INIT(FL_DS, FILEPATH)
        CLASS(FL_DS_TYPE), INTENT(INOUT) :: FL_DS
        CHARACTER(LEN=*),  INTENT(IN)    :: FILEPATH
        INTEGER  :: STATUS

        IF (FL_DS%IS_OPEN) THEN
            STATUS = NF90_CLOSE(FL_DS%FL_NCID)
            CALL NC_CHECK(STATUS, "NF90_CLOSE(FL) IN FL_INIT")
        END IF

        STATUS = NF90_OPEN(TRIM(FILEPATH), NF90_NOWRITE, FL_DS%FL_NCID)
        CALL NC_CHECK(STATUS, "NF90_OPEN(FL)")

        FL_DS%IS_OPEN = .TRUE.

        ! DIMENSION IDs AND LENGTHS
        STATUS = NF90_INQ_DIMID(FL_DS%FL_NCID, "seg_id", FL_DS%DIMID_SEG_ID)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(seg_id)")

        STATUS = NF90_INQUIRE_DIMENSION(FL_DS%FL_NCID, FL_DS%DIMID_SEG_ID, LEN=FL_DS%NSEG)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(seg_id)")

        ! VAR IDs
        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "flight_id", FL_DS%VARID_FL_ID)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(flight_id)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "waypoint", FL_DS%VARID_WP)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(waypoint)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "time_rel_s", FL_DS%VARID_TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_rel_s)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "time_idx", FL_DS%VARID_TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_idx)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "latitude", FL_DS%VARID_LATITUDE)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(latitude)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "longitude", FL_DS%VARID_LONGITUDE)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(longitude)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "level", FL_DS%VARID_LEVEL)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(level)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "altitude", FL_DS%VARID_ALTITUDE)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(altitude)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "true_airspeed", FL_DS%VARID_TRUE_AIRSPEED)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(true_airspeed)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "aircraft_mass", FL_DS%VARID_AIRCRAFT_MASS)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(aircraft_mass)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "engine_efficiency", FL_DS%VARID_ENGINE_EFFICIENCY)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(engine_efficiency)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "fuel_flow", FL_DS%VARID_FUEL_FLOW)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(fuel_flow)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "fuel_burn", FL_DS%VARID_FUEL_BURN)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(fuel_burn)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "thrust", FL_DS%VARID_THRUST)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(thrust)")

        STATUS = NF90_INQ_VARID(FL_DS%FL_NCID, "rocd", FL_DS%VARID_ROCD)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(rocd)")

    END SUBROUTINE FL_INIT

    SUBROUTINE FL_READ_STATIC(FL_DS)
        CLASS(FL_DS_TYPE), INTENT(INOUT) :: FL_DS
        INTEGER :: STATUS

        IF (.NOT. FL_DS%IS_OPEN) STOP "FL_READ_STATIC: FILE NOT OPEN (CALL INIT FIRST)"
        IF (FL_DS%NSEG <= 0) STOP "FL_READ_STATIC: NSEG NOT SET"
        
        IF (.NOT. ALLOCATED(FL_DS%FL_ID)) ALLOCATE(FL_DS%FL_ID(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%WP)) ALLOCATE(FL_DS%WP(FL_DS%NSEG))

        IF (.NOT. ALLOCATED(FL_DS%LATITUDE)) ALLOCATE(FL_DS%LATITUDE(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%LONGITUDE)) ALLOCATE(FL_DS%LONGITUDE(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%LEVEL)) ALLOCATE(FL_DS%LEVEL(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%ALTITUDE)) ALLOCATE(FL_DS%ALTITUDE(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%TIME_REL_S)) ALLOCATE(FL_DS%TIME_REL_S(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%TIME_IDX)) ALLOCATE(FL_DS%TIME_IDX(FL_DS%NSEG))

        IF (.NOT. ALLOCATED(FL_DS%TRUE_AIRSPEED)) ALLOCATE(FL_DS%TRUE_AIRSPEED(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%AIRCRAFT_MASS)) ALLOCATE(FL_DS%AIRCRAFT_MASS(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%ENGINE_EFFICIENCY)) ALLOCATE(FL_DS%ENGINE_EFFICIENCY(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%FUEL_FLOW)) ALLOCATE(FL_DS%FUEL_FLOW(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%FUEL_BURN)) ALLOCATE(FL_DS%FUEL_BURN(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%THRUST)) ALLOCATE(FL_DS%THRUST(FL_DS%NSEG))
        IF (.NOT. ALLOCATED(FL_DS%ROCD)) ALLOCATE(FL_DS%ROCD(FL_DS%NSEG))

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_FL_ID, FL_DS%FL_ID)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(flight_id)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_WP, FL_DS%WP)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(waypoint)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_LATITUDE, FL_DS%LATITUDE)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(latitude)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_LONGITUDE, FL_DS%LONGITUDE)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(longitude)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_LEVEL, FL_DS%LEVEL)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(level)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_ALTITUDE, FL_DS%ALTITUDE)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(altitude)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_TIME_REL_S, FL_DS%TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_rel_s)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_TIME_IDX, FL_DS%TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_idx)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_TRUE_AIRSPEED, FL_DS%TRUE_AIRSPEED)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(true_airspeed)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_AIRCRAFT_MASS, FL_DS%AIRCRAFT_MASS)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(aircraft_mass)")    

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_ENGINE_EFFICIENCY, FL_DS%ENGINE_EFFICIENCY)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(engine_efficiency)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_FUEL_FLOW, FL_DS%FUEL_FLOW)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(fuel_flow)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_FUEL_BURN, FL_DS%FUEL_BURN)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(fuel_burn)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_THRUST, FL_DS%THRUST)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(thrust)")

        STATUS = NF90_GET_VAR(FL_DS%FL_NCID, FL_DS%VARID_ROCD, FL_DS%ROCD)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(rocd)")

    END SUBROUTINE FL_READ_STATIC

    SUBROUTINE FL_SUMMARY(FL_DS)
        CLASS(FL_DS_TYPE), INTENT(IN) :: FL_DS

        PRINT *, "FL_DS SUMMARY:"
        PRINT *, "  NSEG = ", FL_DS%NSEG
        PRINT *, "  IS_OPEN = ", FL_DS%IS_OPEN
        PRINT *, "  NCID = ", FL_DS%FL_NCID
    END SUBROUTINE FL_SUMMARY

    SUBROUTINE FL_CLOSE(FL_DS)
        CLASS(FL_DS_TYPE), INTENT(INOUT) :: FL_DS
        INTEGER :: STATUS

        IF (FL_DS%IS_OPEN) THEN
        STATUS = NF90_CLOSE(FL_DS%FL_NCID)
        CALL NC_CHECK(STATUS, "NF90_CLOSE(FL)")
        END IF

        FL_DS%FL_NCID = -1

        ! FL DIMS
        FL_DS%DIMID_SEG_ID = -1

        FL_DS%NSEG = 0

        ! FL COORDS
        FL_DS%VARID_FL_ID = -1
        FL_DS%VARID_WP = -1

        ! FL VARS
        FL_DS%VARID_TIME_REL_S = -1
        FL_DS%VARID_TIME_IDX = -1
        FL_DS%VARID_LONGITUDE = -1
        FL_DS%VARID_LATITUDE = -1
        FL_DS%VARID_LEVEL = -1
        FL_DS%VARID_ALTITUDE = -1

        FL_DS%VARID_TRUE_AIRSPEED = -1
        FL_DS%VARID_AIRCRAFT_MASS = -1
        FL_DS%VARID_ENGINE_EFFICIENCY = -1
        FL_DS%VARID_FUEL_FLOW = -1
        FL_DS%VARID_FUEL_BURN = -1
        FL_DS%VARID_THRUST = -1
        FL_DS%VARID_ROCD = -1 

        FL_DS%IS_OPEN = .FALSE.

    END SUBROUTINE FL_CLOSE

    SUBROUTINE FL_FINALIZE(FL_DS)
        TYPE(FL_DS_TYPE), INTENT(INOUT) :: FL_DS
        CALL FL_DS%CLOSE()
    END SUBROUTINE FL_FINALIZE

    ! ---------- PL DS METHODS ----------

    SUBROUTINE PL_INIT(PL_DS, FILEPATH)
        CLASS(PL_DS_TYPE), INTENT(INOUT) :: PL_DS
        CHARACTER(LEN=*),  INTENT(IN)    :: FILEPATH

        INTEGER  :: STATUS

        IF (PL_DS%IS_OPEN) THEN
            STATUS = NF90_CLOSE(PL_DS%PL_NCID)
            CALL NC_CHECK(STATUS, "NF90_CLOSE(PL) IN PL_INIT")
        END IF

        STATUS = NF90_OPEN(TRIM(FILEPATH), NF90_NOWRITE, PL_DS%PL_NCID)
        CALL NC_CHECK(STATUS, "NF90_OPEN(PL)")

        PL_DS%IS_OPEN = .TRUE.

        ! DIMENSION IDs AND LENGTHS
        STATUS = NF90_INQ_DIMID(PL_DS%PL_NCID, "seg_id", PL_DS%DIMID_SEG_ID)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(seg_id)")

        STATUS = NF90_INQUIRE_DIMENSION(PL_DS%PL_NCID, PL_DS%DIMID_SEG_ID, LEN=PL_DS%NSEG)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(seg_id)")

        STATUS = NF90_INQ_DIMID(PL_DS%PL_NCID, "species_emi", PL_DS%DIMID_SPECIES_EMI)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(species_emi)")

        STATUS = NF90_INQUIRE_DIMENSION(PL_DS%PL_NCID, PL_DS%DIMID_SPECIES_EMI, LEN=PL_DS%NSEMI)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(species_emi)")
        
        STATUS = NF90_INQ_DIMID(PL_DS%PL_NCID, "time", PL_DS%DIMID_TIME)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(time)")

        STATUS = NF90_INQUIRE_DIMENSION(PL_DS%PL_NCID, PL_DS%DIMID_TIME, LEN=PL_DS%NTPL)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(time)")

        ! VAR IDs
        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "flight_id", PL_DS%VARID_FL_ID)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(flight_id)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "waypoint", PL_DS%VARID_WP)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(waypoint)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "time_rel_s", PL_DS%VARID_TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_rel_s)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "time_idx", PL_DS%VARID_TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_idx)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "species_emi_num", PL_DS%VARID_SPECIES_EMI_NUM)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(species_emi_num)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "latitude", PL_DS%VARID_LATITUDE)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(latitude)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "latitude_m", PL_DS%VARID_LATITUDE_M)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(latitude_m)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "longitude", PL_DS%VARID_LONGITUDE)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(longitude)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "longitude_m", PL_DS%VARID_LONGITUDE_M)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(longitude_m)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "level", PL_DS%VARID_LEVEL)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(level)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "altitude", PL_DS%VARID_ALTITUDE)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(altitude)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "active_seg_flag", PL_DS%VARID_ACTIVE_SEG_FLAG)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(active_seg_flag)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "age_s", PL_DS%VARID_AGE_S)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(age_s)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "width", PL_DS%VARID_WIDTH)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(width)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "depth", PL_DS%VARID_DEPTH)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(depth)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "heading", PL_DS%VARID_HEADING)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(heading)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "sigma_yy", PL_DS%VARID_SIGMA_YY)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(sigma_yy)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "sigma_yz", PL_DS%VARID_SIGMA_YZ)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(sigma_yz)")
        
        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "sigma_zz", PL_DS%VARID_SIGMA_ZZ)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(sigma_zz)")

        STATUS = NF90_INQ_VARID(PL_DS%PL_NCID, "emi_pl_mass", PL_DS%VARID_EMI_PL_MASS)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(emi_pl_mass)")
        
    END SUBROUTINE PL_INIT

    SUBROUTINE PL_READ_STATIC(PL_DS)
        CLASS(PL_DS_TYPE), INTENT(INOUT) :: PL_DS
        INTEGER :: STATUS
        INTEGER :: I, SEG_ID, PL_I
        INTEGER, ALLOCATABLE :: AGE_S_TMP(:,:), ACTIVE_SEG_FLAG_TMP(:,:)
        REAL(DP), ALLOCATABLE :: EMI_PL_MASS_TMP(:,:)
        REAL(DP), ALLOCATABLE :: LATITUDE_TMP(:,:), LONGITUDE_TMP(:,:), LEVEL_TMP(:,:), ALTITUDE_TMP(:,:)
        REAL(DP), ALLOCATABLE :: LATITUDE_M_TMP(:,:), LONGITUDE_M_TMP(:,:)
        REAL(DP), ALLOCATABLE :: WIDTH_TMP(:,:), DEPTH_TMP(:,:), HEADING_TMP(:,:)
        REAL(DP), ALLOCATABLE :: SIGMA_YY_TMP(:,:), SIGMA_YZ_TMP(:,:), SIGMA_ZZ_TMP(:,:)

        IF (.NOT. PL_DS%IS_OPEN) STOP "PL_READ_STATIC: FILE NOT OPEN (CALL INIT FIRST)"
        IF (PL_DS%NSEG <= 0) STOP "PL_READ_STATIC: NSEG NOT SET"
        IF (PL_DS%NSEMI <= 0) STOP "PL_READ_STATIC: NSEMI NOT SET"
        IF (PL_DS%NTPL <= 0) STOP "PL_READ_STATIC: NTPL NOT SET"

        IF (.NOT. ALLOCATED(PL_DS%FL_ID)) ALLOCATE(PL_DS%FL_ID(PL_DS%NSEG))
        IF (.NOT. ALLOCATED(PL_DS%WP)) ALLOCATE(PL_DS%WP(PL_DS%NSEG))        
        
        IF (.NOT. ALLOCATED(PL_DS%TIME_REL_S)) ALLOCATE(PL_DS%TIME_REL_S(PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%TIME_IDX)) ALLOCATE(PL_DS%TIME_IDX(PL_DS%NTPL))

        IF (.NOT. ALLOCATED(PL_DS%SPECIES_EMI_NUM)) ALLOCATE(PL_DS%SPECIES_EMI_NUM(PL_DS%NSEMI))

        IF (.NOT. ALLOCATED(PL_DS%ACTIVE_SEG_FLAG)) ALLOCATE(PL_DS%ACTIVE_SEG_FLAG(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%AGE_S)) ALLOCATE(PL_DS%AGE_S(PL_DS%NSEG, PL_DS%NTPL))

        IF (.NOT. ALLOCATED(PL_DS%LATITUDE)) ALLOCATE(PL_DS%LATITUDE(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%LATITUDE_M)) ALLOCATE(PL_DS%LATITUDE_M(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%LONGITUDE)) ALLOCATE(PL_DS%LONGITUDE(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%LONGITUDE_M)) ALLOCATE(PL_DS%LONGITUDE_M(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%LEVEL)) ALLOCATE(PL_DS%LEVEL(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%ALTITUDE)) ALLOCATE(PL_DS%ALTITUDE(PL_DS%NSEG, PL_DS%NTPL))

        IF (.NOT. ALLOCATED(PL_DS%WIDTH)) ALLOCATE(PL_DS%WIDTH(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%DEPTH)) ALLOCATE(PL_DS%DEPTH(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%HEADING)) ALLOCATE(PL_DS%HEADING(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%SIGMA_YY)) ALLOCATE(PL_DS%SIGMA_YY(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%SIGMA_YZ)) ALLOCATE(PL_DS%SIGMA_YZ(PL_DS%NSEG, PL_DS%NTPL))
        IF (.NOT. ALLOCATED(PL_DS%SIGMA_ZZ)) ALLOCATE(PL_DS%SIGMA_ZZ(PL_DS%NSEG, PL_DS%NTPL))

        IF (.NOT. ALLOCATED(PL_DS%EMI_PL_MASS)) ALLOCATE(PL_DS%EMI_PL_MASS(PL_DS%NSEG, PL_DS%NSEMI))
        
        ! PL COORDS
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_FL_ID, PL_DS%FL_ID)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(flight_id)")

        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_WP, PL_DS%WP)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(waypoint)")
        
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_TIME_REL_S, PL_DS%TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_rel_s)")

        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_TIME_IDX, PL_DS%TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_idx)")

        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_SPECIES_EMI_NUM, PL_DS%SPECIES_EMI_NUM)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(species_emi_num)")
        
        ! PL VARS
        IF (.NOT. ALLOCATED(ACTIVE_SEG_FLAG_TMP)) ALLOCATE(ACTIVE_SEG_FLAG_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_ACTIVE_SEG_FLAG, ACTIVE_SEG_FLAG_TMP, &
                              START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(active_seg_flag)")
        PL_DS%ACTIVE_SEG_FLAG = TRANSPOSE(ACTIVE_SEG_FLAG_TMP)
        DEALLOCATE(ACTIVE_SEG_FLAG_TMP)

        IF (.NOT. ALLOCATED(AGE_S_TMP)) ALLOCATE(AGE_S_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_AGE_S, AGE_S_TMP, &
            START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(age_s)")
        PL_DS%AGE_S = TRANSPOSE(AGE_S_TMP)
        DEALLOCATE(AGE_S_TMP)

        IF (.NOT. ALLOCATED(LATITUDE_TMP)) ALLOCATE(LATITUDE_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_LATITUDE, LATITUDE_TMP, &
            START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(latitude)")

        IF (.NOT. ALLOCATED(LATITUDE_M_TMP)) ALLOCATE(LATITUDE_M_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_LATITUDE_M, LATITUDE_M_TMP, &
            START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(latitude_m)")

        IF (.NOT. ALLOCATED(LONGITUDE_TMP)) ALLOCATE(LONGITUDE_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_LONGITUDE, LONGITUDE_TMP, &
            START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(longitude)")

        IF (.NOT. ALLOCATED(LONGITUDE_M_TMP)) ALLOCATE(LONGITUDE_M_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_LONGITUDE_M, LONGITUDE_M_TMP, &
            START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(longitude_m)")

        IF (.NOT. ALLOCATED(LEVEL_TMP)) ALLOCATE(LEVEL_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_LEVEL, LEVEL_TMP, &
                              START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(level)")

        IF (.NOT. ALLOCATED(ALTITUDE_TMP)) ALLOCATE(ALTITUDE_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_ALTITUDE, ALTITUDE_TMP, &
                              START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(altitude)")

        IF (.NOT. ALLOCATED(WIDTH_TMP)) ALLOCATE(WIDTH_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_WIDTH, WIDTH_TMP, &
                              START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(width)")

        IF (.NOT. ALLOCATED(DEPTH_TMP)) ALLOCATE(DEPTH_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_DEPTH, DEPTH_TMP, &
                              START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(depth)")

        IF (.NOT. ALLOCATED(HEADING_TMP)) ALLOCATE(HEADING_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_HEADING, HEADING_TMP, &
                              START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(heading)")

        IF (.NOT. ALLOCATED(SIGMA_YY_TMP)) ALLOCATE(SIGMA_YY_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_SIGMA_YY, SIGMA_YY_TMP, &
                              START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(sigma_yy)")

        IF (.NOT. ALLOCATED(SIGMA_YZ_TMP)) ALLOCATE(SIGMA_YZ_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_SIGMA_YZ, SIGMA_YZ_TMP, &
                              START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(sigma_yz)")

        IF (.NOT. ALLOCATED(SIGMA_ZZ_TMP)) ALLOCATE(SIGMA_ZZ_TMP(PL_DS%NTPL, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_SIGMA_ZZ, SIGMA_ZZ_TMP, &
                              START=[1, 1], COUNT=[PL_DS%NTPL, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(sigma_zz)")
        
        DO SEG_ID = 1, PL_DS%NSEG
            DO PL_I = 1, PL_DS%NTPL
                PL_DS%LATITUDE(SEG_ID, PL_I) = LATITUDE_TMP(PL_I, SEG_ID)
                PL_DS%LATITUDE_M(SEG_ID, PL_I) = LATITUDE_M_TMP(PL_I, SEG_ID)
                PL_DS%LONGITUDE(SEG_ID, PL_I) = LONGITUDE_TMP(PL_I, SEG_ID)
                PL_DS%LONGITUDE_M(SEG_ID, PL_I) = LONGITUDE_M_TMP(PL_I, SEG_ID)
                PL_DS%LEVEL(SEG_ID, PL_I) = LEVEL_TMP(PL_I, SEG_ID)
                PL_DS%ALTITUDE(SEG_ID, PL_I) = ALTITUDE_TMP(PL_I, SEG_ID)
                PL_DS%WIDTH(SEG_ID, PL_I) = WIDTH_TMP(PL_I, SEG_ID)
                PL_DS%DEPTH(SEG_ID, PL_I) = DEPTH_TMP(PL_I, SEG_ID)
                PL_DS%HEADING(SEG_ID, PL_I) = HEADING_TMP(PL_I, SEG_ID)
                PL_DS%SIGMA_YY(SEG_ID, PL_I) = SIGMA_YY_TMP(PL_I, SEG_ID)
                PL_DS%SIGMA_YZ(SEG_ID, PL_I) = SIGMA_YZ_TMP(PL_I, SEG_ID)
                PL_DS%SIGMA_ZZ(SEG_ID, PL_I) = SIGMA_ZZ_TMP(PL_I, SEG_ID)
            END DO
        END DO
        DEALLOCATE(LATITUDE_TMP)
        DEALLOCATE(LATITUDE_M_TMP)
        DEALLOCATE(LONGITUDE_TMP)
        DEALLOCATE(LONGITUDE_M_TMP)
        DEALLOCATE(LEVEL_TMP)
        DEALLOCATE(ALTITUDE_TMP)
        DEALLOCATE(WIDTH_TMP)
        DEALLOCATE(DEPTH_TMP)
        DEALLOCATE(HEADING_TMP)
        DEALLOCATE(SIGMA_YY_TMP)
        DEALLOCATE(SIGMA_YZ_TMP)
        DEALLOCATE(SIGMA_ZZ_TMP)

        IF (.NOT. ALLOCATED(EMI_PL_MASS_TMP)) ALLOCATE(EMI_PL_MASS_TMP(PL_DS%NSEMI, PL_DS%NSEG))
        STATUS = NF90_GET_VAR(PL_DS%PL_NCID, PL_DS%VARID_EMI_PL_MASS, EMI_PL_MASS_TMP, &
              START=[1, 1], COUNT=[PL_DS%NSEMI, PL_DS%NSEG])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(emi_pl_mass)")
        PL_DS%EMI_PL_MASS = TRANSPOSE(EMI_PL_MASS_TMP)
        DEALLOCATE(EMI_PL_MASS_TMP)

        ! PL ATTRIBUTES
        STATUS = NF90_GET_ATT(PL_DS%PL_NCID, NF90_GLOBAL, "n_slices", PL_DS%NSLICES)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(n_slices)")

        STATUS = NF90_GET_ATT(PL_DS%PL_NCID, NF90_GLOBAL, "f_max", PL_DS%FMAX)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(f_max)")

        STATUS = NF90_GET_ATT(PL_DS%PL_NCID, NF90_GLOBAL, "output_pl_slices", PL_DS%OUTPUT_PL_SLICES)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(output_pl_slices)")

        STATUS = NF90_GET_ATT(PL_DS%PL_NCID, NF90_GLOBAL, "n_points", PL_DS%NPOINTS)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(n_points)")

        STATUS = NF90_GET_ATT(PL_DS%PL_NCID, NF90_GLOBAL, "max_age_s", PL_DS%MAX_AGE_S)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(max_age_s)")

        PRINT *, PL_DS%MAX_AGE_S

    END SUBROUTINE PL_READ_STATIC

    SUBROUTINE PL_SUMMARY(PL_DS)
        CLASS(PL_DS_TYPE), INTENT(IN) :: PL_DS
        PRINT *, "PL_DS SUMMARY:"
        PRINT *, "  NSEG = ", PL_DS%NSEG
        PRINT *, "  NSEMI = ", PL_DS%NSEMI
        PRINT *, "  NTPL = ", PL_DS%NTPL
        PRINT *, "  IS_OPEN = ", PL_DS%IS_OPEN
        PRINT *, "  NCID = ", PL_DS%PL_NCID

        PRINT *, "  NSLICES = ", PL_DS%NSLICES
        PRINT *, "  FMAX = ", PL_DS%FMAX
        PRINT *, "  OUTPUT_PL_SLICES = ", PL_DS%OUTPUT_PL_SLICES
        PRINT *, "  MAX_AGE_S = ", PL_DS%MAX_AGE_S
    END SUBROUTINE PL_SUMMARY

    SUBROUTINE PL_CLOSE(PL_DS)
        CLASS(PL_DS_TYPE), INTENT(INOUT) :: PL_DS
        INTEGER :: STATUS

        IF (PL_DS%IS_OPEN) THEN
        STATUS = NF90_CLOSE(PL_DS%PL_NCID)
        CALL NC_CHECK(STATUS, "NF90_CLOSE(PL)")
        END IF

        PL_DS%PL_NCID = -1

        ! PL DIMS
        PL_DS%DIMID_SEG_ID = -1
        PL_DS%DIMID_SPECIES_EMI = -1
        PL_DS%DIMID_TIME = -1
        
        PL_DS%NSEG = 0
        PL_DS%NSEMI = 0
        PL_DS%NTPL = 0

        ! PL COORDS
        PL_DS%VARID_FL_ID = -1
        PL_DS%VARID_WP = -1
        PL_DS%VARID_TIME_REL_S = -1
        PL_DS%VARID_TIME_IDX = -1
        PL_DS%VARID_SPECIES_EMI_NUM = -1

        ! PL VARS
        PL_DS%VARID_LONGITUDE = -1
        PL_DS%VARID_LONGITUDE_M = -1
        PL_DS%VARID_LATITUDE = -1
        PL_DS%VARID_LATITUDE_M = -1
        PL_DS%VARID_LEVEL = -1
        PL_DS%VARID_ALTITUDE = -1
        PL_DS%VARID_ACTIVE_SEG_FLAG = -1
        PL_DS%VARID_AGE_S = -1
        PL_DS%VARID_WIDTH = -1
        PL_DS%VARID_DEPTH = -1
        PL_DS%VARID_HEADING = -1
        PL_DS%VARID_SIGMA_YY = -1
        PL_DS%VARID_SIGMA_YZ = -1
        PL_DS%VARID_SIGMA_ZZ = -1

        PL_DS%VARID_EMI_PL_MASS = -1       

        PL_DS%IS_OPEN = .FALSE.

    END SUBROUTINE PL_CLOSE

    SUBROUTINE PL_FINALIZE(PL_DS)
        TYPE(PL_DS_TYPE), INTENT(INOUT) :: PL_DS
        CALL PL_DS%CLOSE()
    END SUBROUTINE PL_FINALIZE

    ! ---------- BOXM DS METHODS ----------

    SUBROUTINE BOXM_INIT(BOXM_DS, FILEPATH)
        CLASS(BOXM_DS_TYPE), INTENT(INOUT) :: BOXM_DS
        CHARACTER(LEN=*),  INTENT(IN)    :: FILEPATH

        INTEGER  :: STATUS

        IF (BOXM_DS%IS_OPEN) THEN
            STATUS = NF90_CLOSE(BOXM_DS%BOXM_NCID)
            CALL NC_CHECK(STATUS, "NF90_CLOSE(BOXM) IN BOXM_INIT")
        END IF

        STATUS = NF90_OPEN(TRIM(FILEPATH), NF90_NOWRITE, BOXM_DS%BOXM_NCID)
        CALL NC_CHECK(STATUS, "NF90_OPEN(BOXM)")

        BOXM_DS%IS_OPEN = .TRUE.

        ! DIMENSION IDs AND LENGTHS
        STATUS = NF90_INQ_DIMID(BOXM_DS%BOXM_NCID, "cell", BOXM_DS%DIMID_CELL)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(cell)")

        STATUS = NF90_INQUIRE_DIMENSION(BOXM_DS%BOXM_NCID, BOXM_DS%DIMID_CELL, LEN=BOXM_DS%NCELL)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(cell)")

        STATUS = NF90_INQ_DIMID(BOXM_DS%BOXM_NCID, "species_boxm", BOXM_DS%DIMID_SPECIES_BOXM)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(species_boxm)")

        STATUS = NF90_INQUIRE_DIMENSION(BOXM_DS%BOXM_NCID, BOXM_DS%DIMID_SPECIES_BOXM, LEN=BOXM_DS%NSBOXM)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(species_boxm)")

        STATUS = NF90_INQ_DIMID(BOXM_DS%BOXM_NCID, "time", BOXM_DS%DIMID_TIME)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(time)")

        STATUS = NF90_INQUIRE_DIMENSION(BOXM_DS%BOXM_NCID, BOXM_DS%DIMID_TIME, LEN=BOXM_DS%NTBOXM)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(time)")

        ! VAR IDs
        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "time_rel_s", BOXM_DS%VARID_TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_rel_s)")

        STATUS= NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "time_idx", BOXM_DS%VARID_TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_idx)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "species_boxm_num", BOXM_DS%VARID_SPECIES_BOXM_NUM)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(species_boxm_num)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "longitude_c", BOXM_DS%VARID_LONGITUDE_C)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(longitude_c)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "latitude_c", BOXM_DS%VARID_LATITUDE_C)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(latitude_c)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "altitude_c", BOXM_DS%VARID_ALTITUDE_C)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(altitude_c)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "level_c", BOXM_DS%VARID_LEVEL_C)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(level_c)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "longitude_c_m", BOXM_DS%VARID_LONGITUDE_C_M)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(longitude_c_m)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "latitude_c_m", BOXM_DS%VARID_LATITUDE_C_M)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(latitude_c_m)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "air_temperature", BOXM_DS%VARID_TEMP)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(air_temperature)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "H2O", BOXM_DS%VARID_H2O)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(H2O)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "M", BOXM_DS%VARID_M)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(M)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "O2", BOXM_DS%VARID_O2)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(O2)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "N2", BOXM_DS%VARID_N2)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(N2)")
    
        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "sza", BOXM_DS%VARID_SZA)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(sza)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "Y_bg_c", BOXM_DS%VARID_Y_BG_C)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(Y_bg_c)")

        STATUS = NF90_INQ_VARID(BOXM_DS%BOXM_NCID, "mol_mass_c", BOXM_DS%VARID_MOL_MASS_C)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(mol_mass_c)")

    END SUBROUTINE BOXM_INIT

    SUBROUTINE BOXM_READ_STATIC(BOXM_DS)
        CLASS(BOXM_DS_TYPE), INTENT(INOUT) :: BOXM_DS
        INTEGER :: STATUS, I, SPECIES_ID

        REAL(DP), ALLOCATABLE :: TEMP_TMP(:,:), H2O_TMP(:,:), M_TMP(:,:), O2_TMP(:,:), N2_TMP(:,:), SZA_TMP(:,:)
        REAL(DP), ALLOCATABLE :: Y_BG_C_TMP(:,:)

        IF (.NOT. BOXM_DS%IS_OPEN) STOP "BOXM_READ_STATIC: FILE NOT OPEN (CALL INIT FIRST)"
        IF (BOXM_DS%NCELL <= 0) STOP "BOXM_READ_STATIC: NCELL NOT SET"
        IF (BOXM_DS%NSBOXM <= 0) STOP "BOXM_READ_STATIC: NSBOXM NOT SET"
        IF (BOXM_DS%NTBOXM <= 0) STOP "BOXM_READ_STATIC: NTBOXM NOT SET"

        IF (.NOT. ALLOCATED(BOXM_DS%TIME_REL_S)) ALLOCATE(BOXM_DS%TIME_REL_S(BOXM_DS%NTBOXM))
        IF (.NOT. ALLOCATED(BOXM_DS%TIME_IDX)) ALLOCATE(BOXM_DS%TIME_IDX(BOXM_DS%NTBOXM))
        IF (.NOT. ALLOCATED(BOXM_DS%SPECIES_BOXM_NUM)) ALLOCATE(BOXM_DS%SPECIES_BOXM_NUM(BOXM_DS%NSBOXM))

        IF (.NOT. ALLOCATED(BOXM_DS%LONGITUDE_C)) ALLOCATE(BOXM_DS%LONGITUDE_C(BOXM_DS%NCELL))
        IF (.NOT. ALLOCATED(BOXM_DS%LATITUDE_C)) ALLOCATE(BOXM_DS%LATITUDE_C(BOXM_DS%NCELL))
        IF (.NOT. ALLOCATED(BOXM_DS%ALTITUDE_C)) ALLOCATE(BOXM_DS%ALTITUDE_C(BOXM_DS%NCELL))
        IF (.NOT. ALLOCATED(BOXM_DS%LEVEL_C)) ALLOCATE(BOXM_DS%LEVEL_C(BOXM_DS%NCELL))

        IF (.NOT. ALLOCATED(BOXM_DS%LONGITUDE_C_M)) ALLOCATE(BOXM_DS%LONGITUDE_C_M(BOXM_DS%NCELL))
        IF (.NOT. ALLOCATED(BOXM_DS%LATITUDE_C_M)) ALLOCATE(BOXM_DS%LATITUDE_C_M(BOXM_DS%NCELL))

        IF (.NOT. ALLOCATED(BOXM_DS%TEMP)) ALLOCATE(BOXM_DS%TEMP(BOXM_DS%NCELL, BOXM_DS%NTBOXM))
        IF (.NOT. ALLOCATED(BOXM_DS%H2O)) ALLOCATE(BOXM_DS%H2O(BOXM_DS%NCELL, BOXM_DS%NTBOXM))
        IF (.NOT. ALLOCATED(BOXM_DS%M)) ALLOCATE(BOXM_DS%M(BOXM_DS%NCELL, BOXM_DS%NTBOXM))
        IF (.NOT. ALLOCATED(BOXM_DS%O2)) ALLOCATE(BOXM_DS%O2(BOXM_DS%NCELL, BOXM_DS%NTBOXM))
        IF (.NOT. ALLOCATED(BOXM_DS%N2)) ALLOCATE(BOXM_DS%N2(BOXM_DS%NCELL, BOXM_DS%NTBOXM))
        IF (.NOT. ALLOCATED(BOXM_DS%SZA)) ALLOCATE(BOXM_DS%SZA(BOXM_DS%NCELL, BOXM_DS%NTBOXM))
        IF (.NOT. ALLOCATED(BOXM_DS%Y_BG_C)) ALLOCATE(BOXM_DS%Y_BG_C(BOXM_DS%NCELL, BOXM_DS%NSBOXM))
        IF (.NOT. ALLOCATED(BOXM_DS%MOL_MASS_C)) ALLOCATE(BOXM_DS%MOL_MASS_C(BOXM_DS%NSBOXM))

        ! BOXM ATTRIBUTES
        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "ts_fl", BOXM_DS%TS_FL)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(ts_fl)")
        
        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "ts_pl", BOXM_DS%TS_PL)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(ts_pl)")
        
        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "ts_sim", BOXM_DS%TS_SIM)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(ts_sim)")
        
        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "ts_out", BOXM_DS%TS_OUT)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(ts_out)")

        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "hres_sim_c", BOXM_DS%HRES_SIM_C)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(hres_sim_c)")
        
        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "vres_sim_c", BOXM_DS%VRES_SIM_C)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(vres_sim_c)")
        
        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "hres_sim_f", BOXM_DS%HRES_SIM_F)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(hres_sim_f)")
        
        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "vres_sim_f", BOXM_DS%VRES_SIM_F)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(vres_sim_f)")

        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "photol_params", BOXM_DS%NPP)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(photol_params)")

        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "photol_coeffs", BOXM_DS%NPC)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(photol_coeffs)")

        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "therm_coeffs", BOXM_DS%NTC)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(therm_coeffs)")

        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "flux_species", BOXM_DS%NFL)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(flux_species)")

        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "run_chem", BOXM_DS%RUN_CHEM)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(run_chem)")

        STATUS = NF90_GET_ATT(BOXM_DS%BOXM_NCID, NF90_GLOBAL, "n_ac", BOXM_DS%N_AC)
        CALL NC_CHECK(STATUS, "NF90_GET_ATT(n_ac)")

        PRINT *, BOXM_DS%N_AC
        print *, BOXM_DS%RUN_CHEM

        ! BOXM COORDS
        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_TIME_REL_S, BOXM_DS%TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_rel_s)")

        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_TIME_IDX, BOXM_DS%TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_idx)")

        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_SPECIES_BOXM_NUM, BOXM_DS%SPECIES_BOXM_NUM)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(species_boxm_num)")

        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_LONGITUDE_C, BOXM_DS%LONGITUDE_C)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(longitude_c)")

        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_LATITUDE_C, BOXM_DS%LATITUDE_C)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(latitude_c)")

        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_LONGITUDE_C_M, BOXM_DS%LONGITUDE_C_M)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(longitude_c_m)")

        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_LATITUDE_C_M, BOXM_DS%LATITUDE_C_M)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(latitude_c_m)")

        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_ALTITUDE_C, BOXM_DS%ALTITUDE_C)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(altitude_c)")

        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_LEVEL_C, BOXM_DS%LEVEL_C)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(level_c)")

        ! BOXM VARS
        ALLOCATE(TEMP_TMP(BOXM_DS%NTBOXM, BOXM_DS%NCELL))
        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_TEMP, TEMP_TMP, &
                              START=[1, 1], COUNT=[BOXM_DS%NTBOXM, BOXM_DS%NCELL])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(air_temperature)")
        BOXM_DS%TEMP(:,:) = TRANSPOSE(TEMP_TMP(:,:))
        DEALLOCATE(TEMP_TMP)

        ALLOCATE(H2O_TMP(BOXM_DS%NTBOXM, BOXM_DS%NCELL))
        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_H2O, H2O_TMP, &
                              START=[1, 1], COUNT=[BOXM_DS%NTBOXM, BOXM_DS%NCELL])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(H2O)")
        BOXM_DS%H2O(:,:) = TRANSPOSE(H2O_TMP(:,:))
        DEALLOCATE(H2O_TMP)

        ALLOCATE(M_TMP(BOXM_DS%NTBOXM, BOXM_DS%NCELL))
        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_M, M_TMP, &
                              START=[1, 1], COUNT=[BOXM_DS%NTBOXM, BOXM_DS%NCELL])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(M)")
        BOXM_DS%M(:,:) = TRANSPOSE(M_TMP(:,:))
        DEALLOCATE(M_TMP)

        ALLOCATE(O2_TMP(BOXM_DS%NTBOXM, BOXM_DS%NCELL))
        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_O2, O2_TMP, &
                              START=[1, 1], COUNT=[BOXM_DS%NTBOXM, BOXM_DS%NCELL])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(O2)")
        BOXM_DS%O2(:,:) = TRANSPOSE(O2_TMP(:,:))
        DEALLOCATE(O2_TMP)
        
        ALLOCATE(N2_TMP(BOXM_DS%NTBOXM, BOXM_DS%NCELL))
        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_N2, N2_TMP, &
                              START=[1, 1], COUNT=[BOXM_DS%NTBOXM, BOXM_DS%NCELL])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(N2)")
        BOXM_DS%N2(:,:) = TRANSPOSE(N2_TMP(:,:))
        DEALLOCATE(N2_TMP)

        ALLOCATE(SZA_TMP(BOXM_DS%NTBOXM, BOXM_DS%NCELL))
        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_SZA, SZA_TMP, &
                              START=[1, 1], COUNT=[BOXM_DS%NTBOXM, BOXM_DS%NCELL])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(sza)")
        BOXM_DS%SZA(:,:) = TRANSPOSE(SZA_TMP(:,:))
        DEALLOCATE(SZA_TMP)

        ALLOCATE(Y_BG_C_TMP(BOXM_DS%NSBOXM, BOXM_DS%NCELL))
        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_Y_BG_C, Y_BG_C_TMP, &
                    START=[1, 1], COUNT=[BOXM_DS%NSBOXM, BOXM_DS%NCELL])
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(Y_bg_c)")
        BOXM_DS%Y_BG_C(:,:) = TRANSPOSE(Y_BG_C_TMP(:,:))
        DEALLOCATE(Y_BG_C_TMP)

        ! CONVERT TO MOL/CM^3 FROM MIXING RATIO
        DO SPECIES_ID = 1, BOXM_DS%NSBOXM
            BOXM_DS%Y_BG_C(:,SPECIES_ID) = BOXM_DS%Y_BG_C(:,SPECIES_ID) * BOXM_DS%M(:,1) / 1.0E+09
        END DO

        STATUS = NF90_GET_VAR(BOXM_DS%BOXM_NCID, BOXM_DS%VARID_MOL_MASS_C, BOXM_DS%MOL_MASS_C)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(mol_mass_c)")

    END SUBROUTINE BOXM_READ_STATIC

    SUBROUTINE BOXM_SUMMARY(BOXM_DS)
        CLASS(BOXM_DS_TYPE), INTENT(IN) :: BOXM_DS
        PRINT *, "BOXM_DS SUMMARY:"
        PRINT *, "  NCELL = ", BOXM_DS%NCELL
        PRINT *, "  NSBOXM = ", BOXM_DS%NSBOXM
        PRINT *, "  NTBOXM = ", BOXM_DS%NTBOXM
        PRINT *, "  IS_OPEN = ", BOXM_DS%IS_OPEN
        PRINT *, "  NCID = ", BOXM_DS%BOXM_NCID

    END SUBROUTINE BOXM_SUMMARY

    SUBROUTINE BOXM_CLOSE(BOXM_DS)
        CLASS(BOXM_DS_TYPE), INTENT(INOUT) :: BOXM_DS
        INTEGER :: STATUS

        IF (BOXM_DS%IS_OPEN) THEN
        STATUS = NF90_CLOSE(BOXM_DS%BOXM_NCID)
        CALL NC_CHECK(STATUS, "NF90_CLOSE(BOXM)")
        END IF

        BOXM_DS%BOXM_NCID = -1

        ! BOXM DIMS
        BOXM_DS%DIMID_CELL = -1
        BOXM_DS%DIMID_SPECIES_BOXM = -1
        BOXM_DS%DIMID_TIME = -1

        BOXM_DS%NCELL = 0
        BOXM_DS%NSBOXM = 0
        BOXM_DS%NTBOXM = 0

        ! BOXM COORDS
        BOXM_DS%VARID_TIME_REL_S = -1
        BOXM_DS%VARID_TIME_IDX = -1
        BOXM_DS%VARID_SPECIES_BOXM_NUM = -1
        BOXM_DS%VARID_LONGITUDE_C = -1
        BOXM_DS%VARID_LATITUDE_C = -1
        BOXM_DS%VARID_ALTITUDE_C = -1
        BOXM_DS%VARID_LEVEL_C = -1

        ! BOXM VARS
        BOXM_DS%VARID_TEMP = -1
        BOXM_DS%VARID_H2O = -1
        BOXM_DS%VARID_M = -1
        BOXM_DS%VARID_O2 = -1
        BOXM_DS%VARID_N2 = -1
        BOXM_DS%VARID_SZA = -1
        BOXM_DS%VARID_Y_BG_C = -1
        BOXM_DS%VARID_MOL_MASS_C = -1

        BOXM_DS%IS_OPEN = .FALSE.

    END SUBROUTINE BOXM_CLOSE

    SUBROUTINE BOXM_FINALIZE(BOXM_DS)
        TYPE(BOXM_DS_TYPE), INTENT(INOUT) :: BOXM_DS
        CALL BOXM_DS%CLOSE()
    END SUBROUTINE BOXM_FINALIZE

END MODULE DEFINE_INPUT_TYPES

MODULE DEFINE_STATE_TYPES
    USE NETCDF
    USE HELPERS
    USE RUN_CHEM_UTILS
    USE DEFINE_INPUT_TYPES
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: PL_STATE_TYPE, BOXM_STATE_TYPE, PATCH_STATE_TYPE

    TYPE :: PL_STATE_TYPE
        
        ! PL STATE DIMS
        INTEGER :: NSEG = 0
        INTEGER :: NSPL = 0

        ! PL SLICES DIMS
        INTEGER :: NSLICES = 0
        INTEGER :: NPOINTS = 32  ! number of points to define slice polygons (for grid mapping and diagnostics)
        REAL(DP) :: FMAX = 0.99

        ! PL STATE COORDS
        INTEGER, ALLOCATABLE :: FL_ID(:)
        INTEGER,   ALLOCATABLE :: WP(:)
        INTEGER, ALLOCATABLE :: SPECIES_PL_NUM(:)
        
        ! TIME TRACKING
        INTEGER :: TIME_REL_S = 0

        ! PL STATE VARS
        INTEGER, ALLOCATABLE :: ACTIVE_SEG_FLAG(:)
        INTEGER, ALLOCATABLE :: AGE_S(:)

        REAL(DP), ALLOCATABLE :: LONGITUDE(:)      ! (NSEG)
        REAL(DP), ALLOCATABLE :: LATITUDE(:)      ! (NSEG)
        REAL(DP), ALLOCATABLE :: ALTITUDE(:)      ! (NSEG)
        REAL(DP), ALLOCATABLE :: LEVEL(:)    ! (NSEG) optional if you use model levels

        REAL(DP), ALLOCATABLE :: LONGITUDE_M(:)      ! (NSEG) projected longitude in metres for grid mapping
        REAL(DP), ALLOCATABLE :: LATITUDE_M(:)      ! (NSEG) projected latitude in metres for grid mapping

        REAL(DP), ALLOCATABLE :: WIDTH(:)    ! (NSEG) m (or sigma_y derived)
        REAL(DP), ALLOCATABLE :: DEPTH(:)    ! (NSEG) m (sigma_z)
        REAL(DP), ALLOCATABLE :: HEADING(:)  ! (NSEG) rad or deg

        REAL(DP), ALLOCATABLE :: SIGMA_YY(:)   ! (NSEG)
        REAL(DP), ALLOCATABLE :: SIGMA_YZ(:)   ! (NSEG)
        REAL(DP), ALLOCATABLE :: SIGMA_ZZ(:)   ! (NSEG)

        REAL(DP), ALLOCATABLE :: PL_MASS(:,:)   ! (NSEG, NSPL) kg

        ! PL OUT VARS FOR GRID MAPPING
        REAL(DP), ALLOCATABLE :: Y_HALF(:,:) ! (NSEG, NSLICES)
        REAL(DP), ALLOCATABLE :: Z_HALF(:,:) ! (NSEG, NSLICES)
        REAL(DP), ALLOCATABLE :: M_FRAC(:)      ! (NSLICES)
        REAL(DP), ALLOCATABLE :: W_SLICE(:) ! (NSLICES)

        REAL(DP), ALLOCATABLE :: ELLIPSES_M(:,:,:) ! (NSEG, NPOINTS, 3) projected PL ellipse points for grid mapping
        REAL(DP), ALLOCATABLE :: SLICE_POLYS_M(:,:,:,:) ! (NSEG, NSLICES, 4, 3) projected PL slice corners for grid mapping
        
        ! Sparse projection map: (seg,slice) → fine cells
        INTEGER :: NNZ_MAP = 0  ! number of nonzero weights
        INTEGER, ALLOCATABLE :: MAP_SEG(:)      ! (NNZ_MAP) source segment
        INTEGER, ALLOCATABLE :: MAP_SLICE(:)    ! (NNZ_MAP) source slice
        INTEGER, ALLOCATABLE :: MAP_CELL_C(:)   ! (NNZ_MAP) coarse cell id
        INTEGER, ALLOCATABLE :: MAP_CELL_F(:)   ! (NNZ_MAP) fine subcell id (local or global)
        REAL(DP), ALLOCATABLE :: MAP_W(:)       ! (NNZ_MAP) combined weight (h×v×w_slice)
        

    CONTAINS
        PROCEDURE, PASS :: INIT_FROM_PL_DS => PL_STATE_INIT_FROM_PL_DS
        PROCEDURE, PASS :: INIT_FROM_PL_OUT => PL_STATE_INIT_FROM_PL_OUT
        PROCEDURE, PASS :: BUILD_ELLIPSES_M => PL_STATE_BUILD_ELLIPSES_M
        PROCEDURE, PASS :: BUILD_SLICE_POLYS_M => PL_STATE_BUILD_SLICE_POLYS_M
        PROCEDURE, PASS :: ADVANCE_GEOM    => PL_STATE_ADVANCE_GEOM
        PROCEDURE, PASS :: BUILD_ACTIVE     => PL_STATE_BUILD_ACTIVE
        PROCEDURE, PASS :: PROJECT_TO_GRID => PL_STATE_PROJECT_TO_GRID
        PROCEDURE, PASS :: EMI_TO_PLUMES => PL_STATE_EMI_TO_PLUMES
        PROCEDURE, PASS :: BACKPROJECT_FROM_GRID => PL_STATE_BACKPROJECT_FROM_GRID

    END TYPE PL_STATE_TYPE

    TYPE :: BOXM_STATE_TYPE
        
        ! BOXM DIMS
        INTEGER :: NCELL = 0
        INTEGER :: NSBOXM = 219

        ! BOXM COORDS
        INTEGER :: TIME_REL_S = 0

        INTEGER, ALLOCATABLE :: SPECIES_BOXM_NUM(:)
        REAL(DP), ALLOCATABLE :: MOL_MASS_C(:)   ! (NSBOXM) kg/mol
        REAL(DP), ALLOCATABLE :: LONGITUDE_C(:)     ! (NCELL)
        REAL(DP), ALLOCATABLE :: LATITUDE_C(:)     ! (NCELL)
        REAL(DP), ALLOCATABLE :: ALTITUDE_C(:)     ! (NCELL)
        REAL(DP), ALLOCATABLE :: LEVEL_C(:)     ! (NCELL) or altitude midpoints

        REAL(DP), ALLOCATABLE :: LONGITUDE_C_M(:)     ! (NCELL) projected longitude in metres for grid mapping
        REAL(DP), ALLOCATABLE :: LATITUDE_C_M(:)     ! (NCELL) projected latitude in metres for grid mapping
        REAL(DP), ALLOCATABLE :: DX_C_M(:)           ! (NCELL) coarse-cell width in projected x [m]
        REAL(DP), ALLOCATABLE :: DY_C_M(:)           ! (NCELL) coarse-cell width in projected y [m]

        REAL(DP), ALLOCATABLE :: TEMP(:)      ! (NCELL) K
        REAL(DP), ALLOCATABLE :: H2O(:)       ! (NCELL) WV CONCS mol/m3
        REAL(DP), ALLOCATABLE :: M(:)  ! (NCELL) NUMBER DENSITY mol/m3
        REAL(DP), ALLOCATABLE :: O2(:)  ! (NCELL) OXYGEN CONCS mol/m3
        REAL(DP), ALLOCATABLE :: N2(:)  ! (NCELL) NITROGEN CONCS mol/m3
        REAL(DP), ALLOCATABLE :: SZA(:)      ! (NCELL) SOLAR ZENITH ANGLE deg

        ! Chemistry state on coarse grid, in concentrations (analysis-friendly)
        REAL(DP), ALLOCATABLE :: Y_BG_C(:,:)    ! (NCELL, NSBOXM) mol/m3
        REAL(DP), ALLOCATABLE :: Y_DEL_C(:,:)   ! (NCELL, NSBOXM) mol/m3
        LOGICAL, ALLOCATABLE :: ACTIVE_FLAG(:)     ! (NCELL)

        INTEGER, ALLOCATABLE :: DTS

    CONTAINS
        PROCEDURE, PASS :: INIT_FROM_BOXM_DS => BOXM_STATE_INIT_FROM_BOXM_DS
        PROCEDURE, PASS :: ADVANCE_MET => BOXM_STATE_ADVANCE_MET
        PROCEDURE, PASS :: RUN_COARSE_BG_CHEM => BOXM_STATE_RUN_COARSE_BG_CHEM
        PROCEDURE, PASS :: RUN_COARSE_DELTA_CHEM => BOXM_STATE_RUN_COARSE_DELTA_CHEM

    END TYPE BOXM_STATE_TYPE

    TYPE :: PATCH_STATE_TYPE

        ! PATCH DIMLENS
        INTEGER :: NROWS = 0
        INTEGER :: NSBOXM = 219

        ! PATCH COORDS
        INTEGER :: ROW_IDX = 0

        INTEGER, ALLOCATABLE :: TIME_REL_S(:)
        INTEGER, ALLOCATABLE :: SPECIES_BOXM_NUM(:)
        INTEGER, ALLOCATABLE :: ROW_CELL_C(:)
        INTEGER, ALLOCATABLE :: ROW_CELL_F(:)

        ! PATCH VARS
        REAL(DP), ALLOCATABLE :: Y_DEL_F(:,:)

    CONTAINS
        PROCEDURE, PASS :: INIT_FROM_BOXM_DS => PATCH_STATE_INIT_FROM_BOXM_DS
        PROCEDURE, PASS :: BUILD_ROWS_FROM_W       => PATCH_STATE_BUILD_ROWS_FROM_W
        PROCEDURE, PASS :: ACCUM_DELTAS_FROM_W       => PATCH_STATE_ACCUM_DELTAS_FROM_W
        PROCEDURE, PASS :: RUN_FINE_DELTA_CHEM       => PATCH_STATE_RUN_FINE_DELTA_CHEM

    END TYPE PATCH_STATE_TYPE
   
CONTAINS
    ! ---------- PL STATE METHODS ----------
    SUBROUTINE PL_STATE_INIT_FROM_PL_DS(PL_STATE, PL_DS)
        CLASS(PL_STATE_TYPE), INTENT(INOUT) :: PL_STATE
        CLASS(PL_DS_TYPE),    INTENT(IN)    :: PL_DS

        INTEGER :: I

        PL_STATE%NSEG = PL_DS%NSEG
        PL_STATE%NSPL = PL_DS%NSEMI

        ! ALLOCATE ARRAYS
        IF (.NOT. ALLOCATED(PL_STATE%FL_ID)) ALLOCATE(PL_STATE%FL_ID(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%WP)) ALLOCATE(PL_STATE%WP(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%SPECIES_PL_NUM)) ALLOCATE(PL_STATE%SPECIES_PL_NUM(PL_STATE%NSPL))

        IF (.NOT. ALLOCATED(PL_STATE%ACTIVE_SEG_FLAG)) ALLOCATE(PL_STATE%ACTIVE_SEG_FLAG(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%AGE_S)) ALLOCATE(PL_STATE%AGE_S(PL_STATE%NSEG))

        IF (.NOT. ALLOCATED(PL_STATE%LONGITUDE)) ALLOCATE(PL_STATE%LONGITUDE(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%LONGITUDE_M)) ALLOCATE(PL_STATE%LONGITUDE_M(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%LATITUDE)) ALLOCATE(PL_STATE%LATITUDE(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%LATITUDE_M)) ALLOCATE(PL_STATE%LATITUDE_M(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%ALTITUDE)) ALLOCATE(PL_STATE%ALTITUDE(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%LEVEL)) ALLOCATE(PL_STATE%LEVEL(PL_STATE%NSEG))

        IF (.NOT. ALLOCATED(PL_STATE%WIDTH)) ALLOCATE(PL_STATE%WIDTH(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%DEPTH)) ALLOCATE(PL_STATE%DEPTH(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%HEADING)) ALLOCATE(PL_STATE%HEADING(PL_STATE%NSEG))

        IF (.NOT. ALLOCATED(PL_STATE%SIGMA_YY)) ALLOCATE(PL_STATE%SIGMA_YY(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%SIGMA_YZ)) ALLOCATE(PL_STATE%SIGMA_YZ(PL_STATE%NSEG))
        IF (.NOT. ALLOCATED(PL_STATE%SIGMA_ZZ)) ALLOCATE(PL_STATE%SIGMA_ZZ(PL_STATE%NSEG))

        IF (.NOT. ALLOCATED(PL_STATE%PL_MASS)) ALLOCATE(PL_STATE%PL_MASS(PL_STATE%NSEG, PL_STATE%NSPL))
        PL_STATE%PL_MASS(:,:) = 0.0_DP

        PL_STATE%FL_ID(:) = PL_DS%FL_ID(:)
        PL_STATE%WP(:) = PL_DS%WP(:)

    END SUBROUTINE PL_STATE_INIT_FROM_PL_DS

    SUBROUTINE PL_STATE_INIT_FROM_PL_OUT(PL_STATE, PL_DS, SEG_ID)
        
        CLASS(PL_STATE_TYPE), INTENT(INOUT) :: PL_STATE
        CLASS(PL_DS_TYPE),    INTENT(IN)    :: PL_DS

        INTEGER :: SLICE_ID, SEG_ID

        PL_STATE%NSLICES = PL_DS%NSLICES
        PL_STATE%NPOINTS = PL_DS%NPOINTS
        PL_STATE%FMAX = PL_DS%FMAX

        IF (.NOT. ALLOCATED(PL_STATE%Y_HALF)) ALLOCATE(PL_STATE%Y_HALF(PL_STATE%NSEG, PL_STATE%NSLICES))
        IF (.NOT. ALLOCATED(PL_STATE%Z_HALF)) ALLOCATE(PL_STATE%Z_HALF(PL_STATE%NSEG, PL_STATE%NSLICES))
        IF (.NOT. ALLOCATED(PL_STATE%W_SLICE)) ALLOCATE(PL_STATE%W_SLICE(PL_STATE%NSLICES))
        IF (.NOT. ALLOCATED(PL_STATE%M_FRAC)) ALLOCATE(PL_STATE%M_FRAC(PL_STATE%NSLICES))

        IF (.NOT. ALLOCATED(PL_STATE%ELLIPSES_M)) ALLOCATE(PL_STATE%ELLIPSES_M(PL_STATE%NSEG, PL_STATE%NPOINTS, 3))
        IF (.NOT. ALLOCATED(PL_STATE%SLICE_POLYS_M)) ALLOCATE(PL_STATE%SLICE_POLYS_M(PL_STATE%NSEG, PL_STATE%NSLICES, 4, 3))

        ! DEFINE CUMULATIVE MASS LADDER
        DO SLICE_ID = 1, PL_STATE%NSLICES
            PL_STATE%M_FRAC(SLICE_ID) = PL_STATE%FMAX * REAL(SLICE_ID, DP) / REAL(PL_STATE%NSLICES, DP)
            PRINT *, "SLICE_ID=", SLICE_ID, " M_FRAC=", PL_STATE%M_FRAC(SLICE_ID)
        END DO

        ! CALC WEIGHTS ARRAY
        PL_STATE%W_SLICE(1) = PL_STATE%M_FRAC(1)
        DO SLICE_ID = 2, PL_STATE%NSLICES
            PL_STATE%W_SLICE(SLICE_ID) = PL_STATE%M_FRAC(SLICE_ID) - PL_STATE%M_FRAC(SLICE_ID-1)
            PRINT *, "SLICE_ID=", SLICE_ID, " W_SLICE=", PL_STATE%W_SLICE(SLICE_ID)
        END DO

        ! OPTIONAL STRICT CONSERVATION ADJUSTMENT
        PL_STATE%W_SLICE(PL_STATE%NSLICES) = PL_STATE%W_SLICE(PL_STATE%NSLICES) + &
            (1.0_DP - PL_STATE%FMAX)

        PL_STATE%SLICE_POLYS_M(:,:,:,:) = 0.0_DP

    END SUBROUTINE PL_STATE_INIT_FROM_PL_OUT

    SUBROUTINE PL_STATE_BUILD_ELLIPSES_M(PL_STATE, SEG_ID, NPOINTS)
        CLASS(PL_STATE_TYPE), INTENT(INOUT) :: PL_STATE
        INTEGER, INTENT(IN) :: SEG_ID, NPOINTS
        INTEGER :: PT_ID
        REAL(DP), PARAMETER :: PI = 3.14159265358979323846_DP
        REAL(DP) :: ANGLE_STEP, ANGLE, HEAD_RAD, X, Y, Z, X_LON, X_LAT
        REAL(DP) :: LON_C_M, LAT_C_M, ALT_C, WIDTH, DEPTH
        REAL(DP) :: COORD(3)

        ANGLE_STEP = 360.0_DP / REAL(NPOINTS, DP)
        HEAD_RAD = PL_STATE%HEADING(SEG_ID) * (PI / 180.0_DP) ! CONVERT TO RADIANS

        DO PT_ID = 1, NPOINTS
            ANGLE = (REAL(PT_ID-1, DP) * ANGLE_STEP) * (PI / 180.0_DP) ! CONVERT TO RADIANS

            LON_C_M = PL_STATE%LONGITUDE_M(SEG_ID)
            LAT_C_M = PL_STATE%LATITUDE_M(SEG_ID)
            ALT_C = PL_STATE%ALTITUDE(SEG_ID)

            WIDTH = PL_STATE%WIDTH(SEG_ID)
            DEPTH = PL_STATE%DEPTH(SEG_ID)

            ! IN PLANE OFFSETS
            X = (WIDTH / 2.0_DP) * COS(ANGLE)
            Z = (DEPTH / 2.0_DP) * SIN(ANGLE)
            Y = 0.0_DP ! ALONG HEADING DIRECTION

            ! ROTATE OFFSETS BY HEADING
            X_LON = -X * SIN(HEAD_RAD)
            X_LAT =  X * COS(HEAD_RAD)

            COORD(1) = LON_C_M + X_LON
            COORD(2) = LAT_C_M + X_LAT
            COORD(3) = ALT_C   + Z

            PL_STATE%ELLIPSES_M(SEG_ID, PT_ID, :) = COORD(:)
        END DO

    END SUBROUTINE PL_STATE_BUILD_ELLIPSES_M

    SUBROUTINE PL_STATE_BUILD_SLICE_POLYS_M(PL_STATE, SEG_ID, SLICE_ID)
        CLASS(PL_STATE_TYPE), INTENT(INOUT) :: PL_STATE
        INTEGER, INTENT(IN) :: SEG_ID, SLICE_ID

        REAL(DP), PARAMETER :: EPS_DIR = 1.0E-12_DP
        REAL(DP), PARAMETER :: COS_EPS = 1.0E-6_DP
        REAL(DP), PARAMETER :: MITER_MAX = 4.0_DP

        REAL(DP) :: Y_HALF, Z_HALF
        REAL(DP) :: CENTER(3)
        REAL(DP) :: DIR_PREV(2), DIR_NEXT(2), DIR_USE(2), BIS(2), PERP_BIS(2)
        REAL(DP) :: LEN_PREV, LEN_NEXT, LEN_BIS, LEN_USE
        REAL(DP) :: COS_THETA, MITER_SCALE
        REAL(DP) :: COORD_BL(3), COORD_TL(3), COORD_TR(3), COORD_BR(3)

        INTEGER :: NPREV, NNEXT
        LOGICAL :: HAS_PREV, HAS_NEXT

        ! Get center and half-widths
        CENTER(1) = PL_STATE%LONGITUDE_M(SEG_ID)
        CENTER(2) = PL_STATE%LATITUDE_M(SEG_ID)
        CENTER(3) = PL_STATE%ALTITUDE(SEG_ID)
        Y_HALF = PL_STATE%Y_HALF(SEG_ID, SLICE_ID)
        Z_HALF = PL_STATE%Z_HALF(SEG_ID, SLICE_ID)

        ! Default initializations
        NPREV = SEG_ID
        NNEXT = SEG_ID
        HAS_PREV = .FALSE.
        HAS_NEXT = .FALSE.

        DIR_PREV(:) = 0.0_DP
        DIR_NEXT(:) = 0.0_DP
        DIR_USE(:)  = 0.0_DP
        BIS(:)      = 0.0_DP
        PERP_BIS(:) = 0.0_DP

        ! Only use true geometric neighbors on the same flight / trajectory
        IF (SEG_ID > 1) THEN
            IF (PL_STATE%FL_ID(SEG_ID-1) == PL_STATE%FL_ID(SEG_ID) .AND. &
                PL_STATE%WP(SEG_ID-1)    == PL_STATE%WP(SEG_ID) - 1) THEN
                NPREV = SEG_ID - 1
                HAS_PREV = .TRUE.
            END IF
        END IF

        IF (SEG_ID < PL_STATE%NSEG) THEN
            IF (PL_STATE%FL_ID(SEG_ID+1) == PL_STATE%FL_ID(SEG_ID) .AND. &
                PL_STATE%WP(SEG_ID+1)    == PL_STATE%WP(SEG_ID) + 1) THEN
                NNEXT = SEG_ID + 1
                HAS_NEXT = .TRUE.
            END IF
        END IF

        ! Build local directions in the horizontal plane
        IF (HAS_PREV) THEN
            DIR_PREV(1) = CENTER(1) - PL_STATE%LONGITUDE_M(NPREV)
            DIR_PREV(2) = CENTER(2) - PL_STATE%LATITUDE_M(NPREV)
            LEN_PREV = SQRT(DIR_PREV(1)**2 + DIR_PREV(2)**2)
            IF (LEN_PREV > EPS_DIR) THEN
                DIR_PREV(:) = DIR_PREV(:) / LEN_PREV
            ELSE
                HAS_PREV = .FALSE.
                DIR_PREV(:) = 0.0_DP
            END IF
        END IF

        IF (HAS_NEXT) THEN
            DIR_NEXT(1) = PL_STATE%LONGITUDE_M(NNEXT) - CENTER(1)
            DIR_NEXT(2) = PL_STATE%LATITUDE_M(NNEXT) - CENTER(2)
            LEN_NEXT = SQRT(DIR_NEXT(1)**2 + DIR_NEXT(2)**2)
            IF (LEN_NEXT > EPS_DIR) THEN
                DIR_NEXT(:) = DIR_NEXT(:) / LEN_NEXT
            ELSE
                HAS_NEXT = .FALSE.
                DIR_NEXT(:) = 0.0_DP
            END IF
        END IF

        ! Choose cross-plume orientation direction
        IF (HAS_PREV .AND. HAS_NEXT) THEN
            ! Interior segment: use angle bisector
            BIS(:) = DIR_PREV(:) + DIR_NEXT(:)
            LEN_BIS = SQRT(BIS(1)**2 + BIS(2)**2)

            IF (LEN_BIS > EPS_DIR) THEN
                BIS(:) = BIS(:) / LEN_BIS
            ELSE
                ! Nearly 180-degree reversal or degenerate geometry:
                ! fall back to a single valid local direction.
                BIS(:) = DIR_NEXT(:)
            END IF

            COS_THETA = DIR_PREV(1)*DIR_NEXT(1) + DIR_PREV(2)*DIR_NEXT(2)
            COS_THETA = MAX(MIN(COS_THETA, 1.0_DP), -1.0_DP)

            IF (ABS(COS_THETA - 1.0_DP) < COS_EPS) THEN
                MITER_SCALE = 1.0_DP
            ELSE
                MITER_SCALE = 1.0_DP / COS(0.5_DP * ACOS(COS_THETA))
            END IF

            ! Prevent pathological flare-out at sharp/dirty corners
            MITER_SCALE = MIN(MITER_SCALE, MITER_MAX)

            DIR_USE(:) = BIS(:)

        ELSE IF (HAS_NEXT) THEN
            ! Start/end segment with only forward neighbor
            DIR_USE(:) = DIR_NEXT(:)
            MITER_SCALE = 1.0_DP

        ELSE IF (HAS_PREV) THEN
            ! Start/end segment with only backward neighbor
            DIR_USE(:) = DIR_PREV(:)
            MITER_SCALE = 1.0_DP

        ELSE
            ! Isolated segment fallback: use stored heading if available
            DIR_USE(1) = COS(PL_STATE%HEADING(SEG_ID))
            DIR_USE(2) = SIN(PL_STATE%HEADING(SEG_ID))
            LEN_USE = SQRT(DIR_USE(1)**2 + DIR_USE(2)**2)

            IF (LEN_USE > EPS_DIR) THEN
                DIR_USE(:) = DIR_USE(:) / LEN_USE
            ELSE
                ! Absolute final fallback
                DIR_USE(1) = 1.0_DP
                DIR_USE(2) = 0.0_DP
            END IF

            MITER_SCALE = 1.0_DP
        END IF

        ! Perpendicular horizontal direction gives left/right slice extent
        PERP_BIS(1) = -DIR_USE(2)
        PERP_BIS(2) =  DIR_USE(1)

        ! Bottom Left
        COORD_BL(1) = CENTER(1) - Y_HALF * PERP_BIS(1) * MITER_SCALE
        COORD_BL(2) = CENTER(2) - Y_HALF * PERP_BIS(2) * MITER_SCALE
        COORD_BL(3) = CENTER(3) - Z_HALF

        ! Top Left
        COORD_TL(1) = CENTER(1) - Y_HALF * PERP_BIS(1) * MITER_SCALE
        COORD_TL(2) = CENTER(2) - Y_HALF * PERP_BIS(2) * MITER_SCALE
        COORD_TL(3) = CENTER(3) + Z_HALF

        ! Top Right
        COORD_TR(1) = CENTER(1) + Y_HALF * PERP_BIS(1) * MITER_SCALE
        COORD_TR(2) = CENTER(2) + Y_HALF * PERP_BIS(2) * MITER_SCALE
        COORD_TR(3) = CENTER(3) + Z_HALF

        ! Bottom Right
        COORD_BR(1) = CENTER(1) + Y_HALF * PERP_BIS(1) * MITER_SCALE
        COORD_BR(2) = CENTER(2) + Y_HALF * PERP_BIS(2) * MITER_SCALE
        COORD_BR(3) = CENTER(3) - Z_HALF

        ! Store
        PL_STATE%SLICE_POLYS_M(SEG_ID, SLICE_ID, 1, :) = COORD_BL(:)
        PL_STATE%SLICE_POLYS_M(SEG_ID, SLICE_ID, 2, :) = COORD_TL(:)
        PL_STATE%SLICE_POLYS_M(SEG_ID, SLICE_ID, 3, :) = COORD_TR(:)
        PL_STATE%SLICE_POLYS_M(SEG_ID, SLICE_ID, 4, :) = COORD_BR(:)

    END SUBROUTINE PL_STATE_BUILD_SLICE_POLYS_M

    SUBROUTINE PL_STATE_ADVANCE_GEOM(PL_STATE, PL_DS, BOXM_DS, TIME_IDX)
        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(PL_DS_TYPE),       INTENT(IN)    :: PL_DS
        CLASS(BOXM_DS_TYPE),      INTENT(IN)   :: BOXM_DS

        INTEGER, INTENT(IN) :: TIME_IDX
        INTEGER :: SEG_ID, SLICE_ID, PL_I, I, CORNER_ID
        REAL(DP) :: SIGMA_Y, SIGMA_Z, WIDTH_EXPECTED_FROM_SIGMA_Y, WIDTH_ACTUAL
        REAL(DP) :: F, U
        REAL(DP), PARAMETER :: F_EPS = 1.0E-12_DP

        PL_I = 0
        DO I = 1, PL_DS%NTPL
            IF (PL_DS%TIME_IDX(I) == TIME_IDX) THEN
                PL_I = I
                EXIT
            END IF
        END DO

        IF (PL_I == 0) THEN
            RETURN
        END IF

        ! GRAB PLUME GEOMETRY FROM PL DS
        PL_STATE%LONGITUDE(:) = PL_DS%LONGITUDE(:,PL_I)
        PL_STATE%LATITUDE(:)  = PL_DS%LATITUDE(:,PL_I)
        PL_STATE%LONGITUDE_M(:) = PL_DS%LONGITUDE_M(:,PL_I)
        PL_STATE%LATITUDE_M(:)  = PL_DS%LATITUDE_M(:,PL_I)  
        PL_STATE%ALTITUDE(:)  = PL_DS%ALTITUDE(:,PL_I)
        PL_STATE%LEVEL(:)     = PL_DS%LEVEL(:,PL_I)

        PL_STATE%ACTIVE_SEG_FLAG(:) = PL_DS%ACTIVE_SEG_FLAG(:,PL_I)
        PL_STATE%AGE_S(:)           = PL_DS%AGE_S(:,PL_I)

        PL_STATE%WIDTH(:)     = PL_DS%WIDTH(:,PL_I)
        PL_STATE%DEPTH(:)     = PL_DS%DEPTH(:,PL_I)
        PL_STATE%HEADING(:)   = PL_DS%HEADING(:,PL_I)
        PL_STATE%SIGMA_YY(:)  = PL_DS%SIGMA_YY(:,PL_I)
        PL_STATE%SIGMA_YZ(:)  = PL_DS%SIGMA_YZ(:,PL_I)
        PL_STATE%SIGMA_ZZ(:)  = PL_DS%SIGMA_ZZ(:,PL_I)

        ! BUILD PLUME SEGMENT REPRESENTATION FROM SLICES
        PL_STATE%Y_HALF(:,:) = 0.0_DP
        PL_STATE%Z_HALF(:,:) = 0.0_DP

        DO SEG_ID = 1, PL_STATE%NSEG
            DO SLICE_ID = 1, PL_STATE%NSLICES
                ! SIGMA_YY, SIGMA_ZZ ARE VARIANCES IN YOUR STATE
                SIGMA_Y = SQRT( MAX( PL_STATE%SIGMA_YY(SEG_ID), 1.0E-30_DP ) )
                SIGMA_Z = SQRT( MAX( PL_STATE%SIGMA_ZZ(SEG_ID), 1.0E-30_DP ) )

                ! Y in original order
                F  = PL_STATE%M_FRAC(SLICE_ID)
                F  = MIN( MAX(F, -1.0_DP + F_EPS), 1.0_DP - F_EPS )
                U  = F
                PL_STATE%Y_HALF(SEG_ID, SLICE_ID) = SQRT(2.0_DP) * SIGMA_Y * ERFINV(U)

                ! Z in reverse order
                F  = PL_STATE%M_FRAC(PL_STATE%NSLICES - SLICE_ID + 1)
                ! Z in reverse order
                F  = PL_STATE%M_FRAC(PL_STATE%NSLICES - SLICE_ID + 1)
                F  = MIN( MAX(F, -1.0_DP + F_EPS), 1.0_DP - F_EPS )
                U  = F
                PL_STATE%Z_HALF(SEG_ID, SLICE_ID) = SQRT(2.0_DP) * SIGMA_Z * ERFINV(U)

                ! CALC SLICE POLYS FOR EACH SEGMENT AND SLICE
                CALL PL_STATE%BUILD_ELLIPSES_M(SEG_ID, PL_DS%NPOINTS)
                CALL PL_STATE%BUILD_SLICE_POLYS_M(SEG_ID, SLICE_ID)
            END DO
        END DO
    END SUBROUTINE PL_STATE_ADVANCE_GEOM

    SUBROUTINE PL_STATE_BUILD_ACTIVE(PL_STATE, BOXM_DS, BOXM_STATE)
        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS
        CLASS(BOXM_STATE_TYPE),  INTENT(INOUT) :: BOXM_STATE

        INTEGER :: I, SEG_ID
        LOGICAL :: SEG_IS_INCLUDED
        LOGICAL :: HAS_NEXT_VALID
        REAL(DP) :: X_MIN_PL, X_MAX_PL, Y_MIN_PL, Y_MAX_PL, Z_MIN_PL, Z_MAX_PL
        REAL(DP) :: X_MIN_C, X_MAX_C, Y_MIN_C, Y_MAX_C, Z_MIN_C, Z_MAX_C
        REAL(DP), ALLOCATABLE :: SLICE_BOX_ALL(:,:,:)
        REAL(DP) :: DZ_C, PAD_XY, PAD_Z
        REAL(DP), PARAMETER :: EPS = 1.0E-12_DP
        REAL(DP), PARAMETER :: MASS_EPS = 1.0E-30_DP

        ! RESET BOXM ACTIVE FLAG
        BOXM_STATE%ACTIVE_FLAG(:) = .FALSE.
        DZ_C = BOXM_DS%VRES_SIM_C
        ! Use coarse-scale padding so active-cell envelopes include bridge/extreme
        ! diagonal extents and avoid coarse-grid staircase clipping artifacts.
        PAD_XY = MAX(0.5_DP * BOXM_DS%HRES_SIM_F, BOXM_DS%HRES_SIM_C)
        PAD_Z = MAX(0.5_DP * BOXM_DS%VRES_SIM_F, BOXM_DS%VRES_SIM_C)

        IF (.NOT. ALLOCATED(SLICE_BOX_ALL)) ALLOCATE(SLICE_BOX_ALL(PL_STATE%NSLICES, 8, 3))

        ! Activate coarse cells by segment envelope overlap only.
        DO SEG_ID = 1, PL_STATE%NSEG
            SEG_IS_INCLUDED = (PL_STATE%ACTIVE_SEG_FLAG(SEG_ID) == 1)
            IF (.NOT. SEG_IS_INCLUDED) CYCLE

            ! Use both back and front faces for all slices (8 corners per slice)
            HAS_NEXT_VALID = .FALSE.
            IF (SEG_ID < PL_STATE%NSEG) THEN
                IF (PL_STATE%FL_ID(SEG_ID+1) == PL_STATE%FL_ID(SEG_ID)) THEN
                    IF (PL_STATE%WP(SEG_ID+1) == PL_STATE%WP(SEG_ID) + 1) THEN
                        HAS_NEXT_VALID = .TRUE.
                    END IF
                END IF
            END IF

            IF (HAS_NEXT_VALID) THEN
                SLICE_BOX_ALL(:,1:4,:) = PL_STATE%SLICE_POLYS_M(SEG_ID, :, :, :)
                SLICE_BOX_ALL(:,5:8,:) = PL_STATE%SLICE_POLYS_M(SEG_ID+1, :, :, :)
                X_MIN_PL = MINVAL(SLICE_BOX_ALL(:,:,1)) - PAD_XY
                X_MAX_PL = MAXVAL(SLICE_BOX_ALL(:,:,1)) + PAD_XY
                Y_MIN_PL = MINVAL(SLICE_BOX_ALL(:,:,2)) - PAD_XY
                Y_MAX_PL = MAXVAL(SLICE_BOX_ALL(:,:,2)) + PAD_XY
                Z_MIN_PL = MINVAL(SLICE_BOX_ALL(:,:,3)) - PAD_Z
                Z_MAX_PL = MAXVAL(SLICE_BOX_ALL(:,:,3)) + PAD_Z
            ELSE
                X_MIN_PL = MINVAL(PL_STATE%SLICE_POLYS_M(SEG_ID, :, :, 1)) - PAD_XY
                X_MAX_PL = MAXVAL(PL_STATE%SLICE_POLYS_M(SEG_ID, :, :, 1)) + PAD_XY
                Y_MIN_PL = MINVAL(PL_STATE%SLICE_POLYS_M(SEG_ID, :, :, 2)) - PAD_XY
                Y_MAX_PL = MAXVAL(PL_STATE%SLICE_POLYS_M(SEG_ID, :, :, 2)) + PAD_XY
                Z_MIN_PL = MINVAL(PL_STATE%SLICE_POLYS_M(SEG_ID, :, :, 3)) - PAD_Z
                Z_MAX_PL = MAXVAL(PL_STATE%SLICE_POLYS_M(SEG_ID, :, :, 3)) + PAD_Z
            END IF

            DO I = 1, BOXM_STATE%NCELL
                X_MIN_C = BOXM_STATE%LONGITUDE_C_M(I) - 0.5_DP * BOXM_STATE%DX_C_M(I)
                X_MAX_C = BOXM_STATE%LONGITUDE_C_M(I) + 0.5_DP * BOXM_STATE%DX_C_M(I)
                Y_MIN_C = BOXM_STATE%LATITUDE_C_M(I) - 0.5_DP * BOXM_STATE%DY_C_M(I)
                Y_MAX_C = BOXM_STATE%LATITUDE_C_M(I) + 0.5_DP * BOXM_STATE%DY_C_M(I)
                Z_MIN_C = BOXM_STATE%ALTITUDE_C(I) - 0.5_DP * DZ_C
                Z_MAX_C = BOXM_STATE%ALTITUDE_C(I) + 0.5_DP * DZ_C

                IF (OVERLAP_1D(X_MIN_PL, X_MAX_PL, X_MIN_C, X_MAX_C) <= EPS) CYCLE
                IF (OVERLAP_1D(Y_MIN_PL, Y_MAX_PL, Y_MIN_C, Y_MAX_C) <= EPS) CYCLE
                IF (OVERLAP_1D(Z_MIN_PL, Z_MAX_PL, Z_MIN_C, Z_MAX_C) <= EPS) CYCLE

                BOXM_STATE%ACTIVE_FLAG(I) = .TRUE.
            END DO
            ! PRINT *, "SEG_ID=", SEG_ID, " ACTIVATES ", COUNT(BOXM_STATE%ACTIVE_FLAG), " CELLS"
        END DO
        DEALLOCATE(SLICE_BOX_ALL)

    END SUBROUTINE PL_STATE_BUILD_ACTIVE

    SUBROUTINE PL_STATE_EMI_TO_PLUMES(PL_STATE, FL_DS, PL_DS, BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(FL_DS_TYPE),       INTENT(IN)    :: FL_DS
        CLASS(PL_DS_TYPE),       INTENT(IN)    :: PL_DS
        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS
        CLASS(BOXM_STATE_TYPE),  INTENT(IN)    :: BOXM_STATE
        CLASS(PATCH_STATE_TYPE), INTENT(IN)    :: PATCH_STATE
        
        INTEGER :: SEG_ID, SLICE_ID, SPECIES_ID, TIME_IDX, I, STATE_I, EMI_ID, PL_ID
        REAL(DP) :: MASS_SEG

        INTEGER, ALLOCATABLE :: EMI_SEG_IDS(:)
        LOGICAL, ALLOCATABLE :: MASK(:)


        IF (.NOT. ALLOCATED(PL_STATE%PL_MASS)) THEN
            ALLOCATE(PL_STATE%PL_MASS(PL_STATE%NSEG, PL_STATE%NSPL))
        END IF

        ! BEFORE PLUME TIME, ZERO OUT MASS
        IF (TIME_IDX < PL_DS%TIME_IDX(1)) THEN
            RETURN
        END IF

        IF ( (TIME_IDX >= PL_DS%TIME_IDX(1)) .AND. (TIME_IDX <= PL_DS%TIME_IDX(PL_DS%NTPL)) ) THEN
            
            DO SEG_ID = 1, PL_STATE%NSEG
                DO PL_ID = 1, PL_STATE%NSPL
                    ! Default: retain previous mass
                    ! Check if segment is too old (expired)
                    IF (PL_STATE%AGE_S(SEG_ID) > PL_DS%MAX_AGE_S) THEN
                        PL_STATE%PL_MASS(SEG_ID, PL_ID) = 0.0_DP
                    END IF
                END DO
            END DO
            
            ! IF EMISSION HAPPENED AT THIS TIMESTEP, SET DATA TO PL MASS
            MASK = (FL_DS%TIME_IDX == TIME_IDX)
            EMI_SEG_IDS = PACK([(I, I=1, SIZE(FL_DS%TIME_IDX))], MASK)

            DO I = 1, SIZE(EMI_SEG_IDS)
                SEG_ID = EMI_SEG_IDS(I)
                DO EMI_ID = 1, PL_DS%NSEMI
                    DO PL_ID = 1, PL_STATE%NSPL
                        IF (PL_STATE%SPECIES_PL_NUM(PL_ID) == PL_DS%SPECIES_EMI_NUM(EMI_ID)) THEN
                            PL_STATE%PL_MASS(SEG_ID, PL_ID) = PL_DS%EMI_PL_MASS(SEG_ID, EMI_ID)
                        END IF
                    END DO
                END DO
            END DO
            IF (ALLOCATED(EMI_SEG_IDS)) DEALLOCATE(EMI_SEG_IDS)
        END IF
    END SUBROUTINE PL_STATE_EMI_TO_PLUMES

    SUBROUTINE PL_STATE_PROJECT_TO_GRID(PL_STATE, PL_DS, BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
            
        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(PATCH_STATE_TYPE),  INTENT(INOUT) :: PATCH_STATE
        CLASS(BOXM_DS_TYPE),      INTENT(IN)    :: BOXM_DS
        CLASS(BOXM_STATE_TYPE),    INTENT(IN)    :: BOXM_STATE
        CLASS(PL_DS_TYPE),       INTENT(IN)    :: PL_DS

        INTEGER :: CELL_C, CELL_F_LOCAL, ROW_IDX, NNZ, TIME_IDX
        INTEGER :: N_ACTIVE_CELL, ACTIVE_IDX
        INTEGER :: NX_F, NY_F, NZ_F, SEG_ID, SLICE_ID, K1, K2
        INTEGER :: FL_S, WP_S, SEG_CAND
        INTEGER :: IX_F, IY_F, IZ_F
        INTEGER :: N_ACTIVE_SLICE, N_NONZERO_SLICE, N_INCLUDED_SEG
        INTEGER, ALLOCATABLE :: ACTIVE_CELLS(:)
        LOGICAL, ALLOCATABLE :: SEG_INCLUDED(:)
        LOGICAL :: HAS_NEXT_VALID
        REAL(DP), ALLOCATABLE :: SEG_X_MIN(:), SEG_X_MAX(:), SEG_Y_MIN(:), SEG_Y_MAX(:), SEG_Z_MIN(:), SEG_Z_MAX(:)
        REAL(DP) :: ALT_C, LON_C_M, LAT_C_M
        REAL(DP) :: DX_C_M, DY_C_M, DZ_C
        REAL(DP) :: DX_F, DY_F, DZ_F
        REAL(DP) :: X_MIN_SEG, X_MAX_SEG, Y_MIN_SEG, Y_MAX_SEG, Z_MIN_SEG, Z_MAX_SEG
        REAL(DP) :: X_MIN_F, X_MAX_F, Y_MIN_F, Y_MAX_F, Z_MIN_F, Z_MAX_F
        REAL(DP) :: X_MIN_C, X_MAX_C, Y_MIN_C, Y_MAX_C, Z_MIN_C, Z_MAX_C
        REAL(DP) :: RAW, RAW_SUM, WEIGHT, RAW_CUR
        REAL(DP) :: SLICE_POLY(4, 3), SLICE_POLY_CUR(4, 3)
        REAL(DP), PARAMETER :: EPS = 1.0E-12_DP
        REAL(DP), PARAMETER :: MASS_EPS = 1.0E-30_DP
        REAL(DP) :: SEG_PAD_XY, SEG_PAD_Z

        REAL(DP), ALLOCATABLE :: RAW_COARSE(:)
        REAL(DP) :: SUM_RAW_COARSE
        REAL(DP), ALLOCATABLE :: RAW_FINE(:,:,:)
        REAL(DP) :: SUM_RAW_FINE, TOTAL_PATCH_Y_DEL_F

        INTEGER :: K, S, SL
        REAL(DP) :: SUM_W

        INTEGER :: SEG_ID_NEXT, I
        LOGICAL :: VALID_SEGMENT
        REAL(DP) :: SLICE_BOX(8,3)
        REAL(DP), ALLOCATABLE :: SLICE_BOX_ALL(:,:,:)

        INTEGER, ALLOCATABLE :: TMP_SEG_ID(:), TMP_SLICE_ID(:), TMP_CELL_C(:), TMP_CELL_F(:)
        REAL(DP), ALLOCATABLE :: TMP_WEIGHT(:)
        INTEGER :: TMP_NNZ, TMP_IDX

        INTEGER, ALLOCATABLE :: VALID_SEG_IDS(:), VALID_SLICE_IDS(:)
        INTEGER :: N_VALID_PAIR, PAIR_IDX

        ! At entry to PROJECTION
        IF (ALLOCATED(PL_STATE%PL_MASS)) THEN
            PRINT *, 'DEBUG: TOTAL PLUME MASS AT ENTRY (PROJECTION) =', SUM(PL_STATE%PL_MASS)
        END IF
        CALL FLUSH(6)

        IF (ALLOCATED(PATCH_STATE%Y_DEL_F)) THEN
            TOTAL_PATCH_Y_DEL_F = 0.0_DP
            DO I = 1, SIZE(PATCH_STATE%Y_DEL_F, 1)
                TOTAL_PATCH_Y_DEL_F = TOTAL_PATCH_Y_DEL_F + PATCH_STATE%Y_DEL_F(I, 8)
            END DO
            PRINT *, 'DEBUG: TOTAL PATCH Y_DEL_F AT ENTRY (PROJECTION) = ', TOTAL_PATCH_Y_DEL_F
            CALL FLUSH(6)
        END IF

        ! CALCULATE FINE GRID SUBDIVISION
        NX_F = MAX(1, NINT(BOXM_DS%HRES_SIM_C / BOXM_DS%HRES_SIM_F)) ! NFINE IN X DIRECTION
        NY_F = NX_F ! MAINTAIN SQUARE CELLS IN HORIZONTAL
        NZ_F = MAX(1, NINT(BOXM_DS%VRES_SIM_C / BOXM_DS%VRES_SIM_F)) ! NFINE IN Z DIRECTION

        DZ_C = BOXM_DS%VRES_SIM_C ! COARSE CELL HEIGHT

        ! DEALLOCATE MAP ARRAYS IF THEY EXIST FROM PREVIOUS CALLS, 
        IF (ALLOCATED(PL_STATE%MAP_SEG)) DEALLOCATE(PL_STATE%MAP_SEG)
        IF (ALLOCATED(PL_STATE%MAP_SLICE)) DEALLOCATE(PL_STATE%MAP_SLICE)
        IF (ALLOCATED(PL_STATE%MAP_CELL_C)) DEALLOCATE(PL_STATE%MAP_CELL_C)
        IF (ALLOCATED(PL_STATE%MAP_CELL_F)) DEALLOCATE(PL_STATE%MAP_CELL_F)
        IF (ALLOCATED(PL_STATE%MAP_W)) DEALLOCATE(PL_STATE%MAP_W)

        N_ACTIVE_SLICE = 0
        N_NONZERO_SLICE = 0

        N_ACTIVE_CELL = COUNT(BOXM_STATE%ACTIVE_FLAG)
        
        PL_STATE%NNZ_MAP = 0
        IF (N_ACTIVE_CELL <= 0) RETURN

        ! THEN ALLOCATE NEW ONES BASED ON CURRENT ACTIVE CELLS AND SEGMENTS
        ALLOCATE(ACTIVE_CELLS(N_ACTIVE_CELL))
        ACTIVE_CELLS(:) = 0

        ACTIVE_IDX = 0
        DO CELL_C = 1, BOXM_STATE%NCELL
            IF (.NOT. BOXM_STATE%ACTIVE_FLAG(CELL_C)) CYCLE
            ACTIVE_IDX = ACTIVE_IDX + 1
            ACTIVE_CELLS(ACTIVE_IDX) = CELL_C
        END DO

        ALLOCATE(SEG_INCLUDED(PL_STATE%NSEG))
        ALLOCATE(SEG_X_MIN(PL_STATE%NSEG), SEG_X_MAX(PL_STATE%NSEG))
        ALLOCATE(SEG_Y_MIN(PL_STATE%NSEG), SEG_Y_MAX(PL_STATE%NSEG))
        ALLOCATE(SEG_Z_MIN(PL_STATE%NSEG), SEG_Z_MAX(PL_STATE%NSEG))
        
        ALLOCATE(SLICE_BOX_ALL(PL_STATE%NSLICES, 8, 3))

        ! First determine which segments to include based on active flags and mass, 
        ! and precompute their bounding boxes with padding
        SEG_INCLUDED(:) = (PL_STATE%ACTIVE_SEG_FLAG(:) == 1)
        SEG_PAD_XY = 0.5_DP * BOXM_DS%HRES_SIM_F
        SEG_PAD_Z = 0.5_DP * BOXM_DS%VRES_SIM_F

        N_INCLUDED_SEG = COUNT(SEG_INCLUDED)

        ! Precompute segment bounding boxes with padding for quick rejection in overlap tests
        DO SEG_ID = 1, PL_STATE%NSEG
            IF (.NOT. SEG_INCLUDED(SEG_ID)) CYCLE
            ! Use both back and front faces for bounding box
            HAS_NEXT_VALID = .FALSE.
            IF (SEG_ID < PL_STATE%NSEG) THEN
                IF (PL_STATE%FL_ID(SEG_ID+1) == PL_STATE%FL_ID(SEG_ID)) THEN
                    IF (PL_STATE%WP(SEG_ID+1) == PL_STATE%WP(SEG_ID) + 1) THEN
                        HAS_NEXT_VALID = .TRUE.
                    END IF
                END IF
            END IF

            IF (HAS_NEXT_VALID) THEN
                SLICE_BOX_ALL(:,1:4,:) = PL_STATE%SLICE_POLYS_M(SEG_ID,:,:,:)
                SLICE_BOX_ALL(:,5:8,:) = PL_STATE%SLICE_POLYS_M(SEG_ID+1,:,:,:)

                SEG_X_MIN(SEG_ID) = MINVAL(SLICE_BOX_ALL(:,:,1)) - SEG_PAD_XY
                SEG_X_MAX(SEG_ID) = MAXVAL(SLICE_BOX_ALL(:,:,1)) + SEG_PAD_XY
                SEG_Y_MIN(SEG_ID) = MINVAL(SLICE_BOX_ALL(:,:,2)) - SEG_PAD_XY
                SEG_Y_MAX(SEG_ID) = MAXVAL(SLICE_BOX_ALL(:,:,2)) + SEG_PAD_XY
                SEG_Z_MIN(SEG_ID) = MINVAL(SLICE_BOX_ALL(:,:,3)) - SEG_PAD_Z
                SEG_Z_MAX(SEG_ID) = MAXVAL(SLICE_BOX_ALL(:,:,3)) + SEG_PAD_Z
            ELSE
                SEG_X_MIN(SEG_ID) = MINVAL(PL_STATE%SLICE_POLYS_M(SEG_ID,:,:,1)) - SEG_PAD_XY
                SEG_X_MAX(SEG_ID) = MAXVAL(PL_STATE%SLICE_POLYS_M(SEG_ID,:,:,1)) + SEG_PAD_XY
                SEG_Y_MIN(SEG_ID) = MINVAL(PL_STATE%SLICE_POLYS_M(SEG_ID,:,:,2)) - SEG_PAD_XY
                SEG_Y_MAX(SEG_ID) = MAXVAL(PL_STATE%SLICE_POLYS_M(SEG_ID,:,:,2)) + SEG_PAD_XY
                SEG_Z_MIN(SEG_ID) = MINVAL(PL_STATE%SLICE_POLYS_M(SEG_ID,:,:,3)) - SEG_PAD_Z
                SEG_Z_MAX(SEG_ID) = MAXVAL(PL_STATE%SLICE_POLYS_M(SEG_ID,:,:,3)) + SEG_PAD_Z
            END IF
        END DO

        ! Pass 1: count all nonzero overlaps so we can allocate the sparse map exactly once.
        ! --- Pass 1: Identify valid (SEG_ID, SLICE_ID) pairs and count NNZ ---
        TMP_NNZ = 0
        ALLOCATE(TMP_SEG_ID(PL_STATE%NSEG*PL_STATE%NSLICES*N_ACTIVE_CELL*NX_F*NY_F*NZ_F))
        ALLOCATE(TMP_SLICE_ID(SIZE(TMP_SEG_ID)))
        ALLOCATE(TMP_CELL_C(SIZE(TMP_SEG_ID)))
        ALLOCATE(TMP_CELL_F(SIZE(TMP_SEG_ID)))
        ALLOCATE(TMP_WEIGHT(SIZE(TMP_SEG_ID)))
        ! PRINT *, 'DEBUG: Starting projection loop, NSEG =', PL_STATE%NSEG, 'NSLICES =', PL_STATE%NSLICES
        DO SEG_ID = 1, PL_STATE%NSEG
            IF (.NOT. SEG_INCLUDED(SEG_ID)) CYCLE
            SEG_ID_NEXT = SEG_ID + 1
            ! Ensure SLICE_POLYS_M is initialized for both SEG_ID and SEG_ID_NEXT
            DO SLICE_ID = 1, PL_STATE%NSLICES
                CALL PL_STATE%BUILD_SLICE_POLYS_M(SEG_ID, SLICE_ID)
                IF (SEG_ID_NEXT <= PL_STATE%NSEG) THEN
                    CALL PL_STATE%BUILD_SLICE_POLYS_M(SEG_ID_NEXT, SLICE_ID)
                END IF
            END DO
            DO SLICE_ID = 1, PL_STATE%NSLICES
                VALID_SEGMENT = .FALSE.
                VALID_SEGMENT = .FALSE.
                IF (SEG_ID_NEXT <= PL_STATE%NSEG) THEN
                    IF (PL_STATE%FL_ID(SEG_ID_NEXT) == PL_STATE%FL_ID(SEG_ID) .AND. &
                        PL_STATE%WP(SEG_ID_NEXT) == PL_STATE%WP(SEG_ID) + 1) THEN
                        VALID_SEGMENT = .TRUE.
                    END IF
                END IF
                IF (.NOT. VALID_SEGMENT) CYCLE
                SLICE_BOX(1:4,:) = PL_STATE%SLICE_POLYS_M(SEG_ID,SLICE_ID,:,:)
                SLICE_BOX(5:8,:) = PL_STATE%SLICE_POLYS_M(SEG_ID_NEXT,SLICE_ID,:,:)
                X_MIN_SEG = MINVAL(SLICE_BOX(:,1))
                X_MAX_SEG = MAXVAL(SLICE_BOX(:,1))
                Y_MIN_SEG = MINVAL(SLICE_BOX(:,2))
                Y_MAX_SEG = MAXVAL(SLICE_BOX(:,2))
                Z_MIN_SEG = MINVAL(SLICE_BOX(:,3))
                Z_MAX_SEG = MAXVAL(SLICE_BOX(:,3))

                ! DO ACTIVE_IDX = 1, N_ACTIVE_CELL
                !     CELL_C = ACTIVE_CELLS(ACTIVE_IDX)
                DO CELL_C = 1, BOXM_STATE%NCELL
                    ALT_C = BOXM_STATE%ALTITUDE_C(CELL_C)
                    LON_C_M = BOXM_STATE%LONGITUDE_C_M(CELL_C)
                    LAT_C_M = BOXM_STATE%LATITUDE_C_M(CELL_C)
                    DX_C_M = BOXM_STATE%DX_C_M(CELL_C)
                    DY_C_M = BOXM_STATE%DY_C_M(CELL_C)
                    DX_F = DX_C_M / REAL(NX_F,DP)
                    DY_F = DY_C_M / REAL(NY_F,DP)
                    DZ_F = DZ_C / REAL(NZ_F,DP)
                    X_MIN_C = LON_C_M - 0.5_DP * DX_C_M
                    X_MAX_C = LON_C_M + 0.5_DP * DX_C_M
                    Y_MIN_C = LAT_C_M - 0.5_DP * DY_C_M
                    Y_MAX_C = LAT_C_M + 0.5_DP * DY_C_M
                    Z_MIN_C = ALT_C - 0.5_DP * DZ_C
                    Z_MAX_C = ALT_C + 0.5_DP * DZ_C
                    IF (OVERLAP_1D(X_MIN_SEG - DX_F, X_MAX_SEG + DX_F, X_MIN_C, X_MAX_C) <= EPS) CYCLE
                    IF (OVERLAP_1D(Y_MIN_SEG - DY_F, Y_MAX_SEG + DY_F, Y_MIN_C, Y_MAX_C) <= EPS) CYCLE
                    IF (OVERLAP_1D(Z_MIN_SEG - DZ_F, Z_MAX_SEG + DZ_F, Z_MIN_C, Z_MAX_C) <= EPS) CYCLE
                    SUM_RAW_FINE = 0.0_DP
                    DO IZ_F = 1, NZ_F
                        Z_MIN_F = ALT_C - 0.5_DP * DZ_C + REAL(IZ_F - 1, DP) * DZ_F
                        Z_MAX_F = Z_MIN_F + DZ_F
                        DO IY_F = 1, NY_F
                            Y_MIN_F = LAT_C_M - 0.5_DP * DY_C_M + REAL(IY_F - 1, DP) * DY_F
                            Y_MAX_F = Y_MIN_F + DY_F
                            DO IX_F = 1, NX_F
                                X_MIN_F = LON_C_M - 0.5_DP * DX_C_M + REAL(IX_F - 1, DP) * DX_F
                                X_MAX_F = X_MIN_F + DX_F
                                RAW = RECT_SLICE_RAW_OVERLAP(X_MIN_F, X_MAX_F, Y_MIN_F, Y_MAX_F, Z_MIN_F, Z_MAX_F, SLICE_BOX)
                                IF (RAW > EPS) THEN
                                    TMP_NNZ = TMP_NNZ + 1
                                    TMP_SEG_ID(TMP_NNZ) = SEG_ID
                                    TMP_SLICE_ID(TMP_NNZ) = SLICE_ID
                                    TMP_CELL_C(TMP_NNZ) = CELL_C
                                    TMP_CELL_F(TMP_NNZ) = (IZ_F - 1) * NX_F * NY_F + (IY_F - 1) * NX_F + IX_F
                                    TMP_WEIGHT(TMP_NNZ) = RAW
                                END IF
                            END DO
                        END DO
                    END DO
                END DO
            END DO
        END DO
        NNZ = TMP_NNZ
        ! PRINT *, 'DEBUG: Finished projection loop, NNZ =', NNZ
        IF (NNZ <= 0) THEN
            PRINT *, 'DEBUG: NNZ <= 0, skipping allocation and returning early.'
            IF (ALLOCATED(ACTIVE_CELLS)) DEALLOCATE(ACTIVE_CELLS)
            IF (ALLOCATED(SEG_INCLUDED)) DEALLOCATE(SEG_INCLUDED)
            IF (ALLOCATED(SEG_X_MIN)) DEALLOCATE(SEG_X_MIN, SEG_X_MAX)
            IF (ALLOCATED(SEG_Y_MIN)) DEALLOCATE(SEG_Y_MIN, SEG_Y_MAX)
            IF (ALLOCATED(SEG_Z_MIN)) DEALLOCATE(SEG_Z_MIN, SEG_Z_MAX)
            RETURN
        END IF
        PL_STATE%NNZ_MAP = NNZ
        ! PRINT *, 'DEBUG: Finished projection loop, NNZ =', NNZ
        IF (NNZ <= 0) THEN
            PRINT *, 'DEBUG: NNZ <= 0, skipping allocation and returning early.'
            IF (ALLOCATED(ACTIVE_CELLS)) DEALLOCATE(ACTIVE_CELLS)
            IF (ALLOCATED(SEG_INCLUDED)) DEALLOCATE(SEG_INCLUDED)
            IF (ALLOCATED(SEG_X_MIN)) DEALLOCATE(SEG_X_MIN, SEG_X_MAX)
            IF (ALLOCATED(SEG_Y_MIN)) DEALLOCATE(SEG_Y_MIN, SEG_Y_MAX)
            IF (ALLOCATED(SEG_Z_MIN)) DEALLOCATE(SEG_Z_MIN, SEG_Z_MAX)
            RETURN
        END IF

        ALLOCATE(PL_STATE%MAP_SEG(NNZ))
        ALLOCATE(PL_STATE%MAP_SLICE(NNZ))
        ALLOCATE(PL_STATE%MAP_CELL_C(NNZ))
        ALLOCATE(PL_STATE%MAP_CELL_F(NNZ))
        ALLOCATE(PL_STATE%MAP_W(NNZ))

        ! Pass 2: for each slice, recompute candidate overlaps, normalize them, and store.
        ROW_IDX = 0
        DO TMP_IDX = 1, NNZ
            ROW_IDX = ROW_IDX + 1
            PL_STATE%MAP_SEG(ROW_IDX) = TMP_SEG_ID(TMP_IDX)
            PL_STATE%MAP_SLICE(ROW_IDX) = TMP_SLICE_ID(TMP_IDX)
            PL_STATE%MAP_CELL_C(ROW_IDX) = TMP_CELL_C(TMP_IDX)
            PL_STATE%MAP_CELL_F(ROW_IDX) = TMP_CELL_F(TMP_IDX)
            PL_STATE%MAP_W(ROW_IDX) = TMP_WEIGHT(TMP_IDX)
        END DO

        ! Normalize weights per (segment, slice)
        DO S = 1, PL_STATE%NSEG
            DO SL = 1, PL_STATE%NSLICES
                SUM_W = 0.0_DP
                DO K = 1, PL_STATE%NNZ_MAP
                    IF (PL_STATE%MAP_SEG(K) == S .AND. PL_STATE%MAP_SLICE(K) == SL) THEN
                        SUM_W = SUM_W + PL_STATE%MAP_W(K)
                    END IF
                END DO

                IF (SUM_W > EPS) THEN
                    DO K = 1, PL_STATE%NNZ_MAP
                        IF (PL_STATE%MAP_SEG(K) == S .AND. PL_STATE%MAP_SLICE(K) == SL) THEN
                            PL_STATE%MAP_W(K) = PL_STATE%W_SLICE(SL) * PL_STATE%MAP_W(K) / SUM_W
                        END IF
                    END DO
                END IF
            END DO
        END DO

        DEALLOCATE(TMP_SEG_ID, TMP_SLICE_ID, TMP_CELL_C, TMP_CELL_F, TMP_WEIGHT)

        ! At exit from PROJECTION
        IF (ALLOCATED(PL_STATE%PL_MASS)) THEN
            PRINT *, 'DEBUG: TOTAL PLUME MASS AT EXIT (PROJECTION) =', SUM(PL_STATE%PL_MASS)
        END IF
        CALL FLUSH(6)

        IF (ALLOCATED(PATCH_STATE%Y_DEL_F)) THEN
            TOTAL_PATCH_Y_DEL_F = 0.0_DP
            DO I = 1, SIZE(PATCH_STATE%Y_DEL_F, 1)
                TOTAL_PATCH_Y_DEL_F = TOTAL_PATCH_Y_DEL_F + PATCH_STATE%Y_DEL_F(I, 8)
            END DO
            PRINT *, 'DEBUG: TOTAL PATCH Y_DEL_F AT EXIT (PROJECTION) = ', TOTAL_PATCH_Y_DEL_F
            CALL FLUSH(6)
        END IF

        IF (ROW_IDX /= NNZ) THEN
            PRINT *, "PL_STATE_PROJECT_TO_GRID: sparse map row mismatch", ROW_IDX, NNZ
            STOP 1
        END IF

        IF (ALLOCATED(ACTIVE_CELLS)) DEALLOCATE(ACTIVE_CELLS)
        IF (ALLOCATED(SEG_INCLUDED)) DEALLOCATE(SEG_INCLUDED)
        IF (ALLOCATED(SEG_X_MIN)) DEALLOCATE(SEG_X_MIN, SEG_X_MAX)
        IF (ALLOCATED(SEG_Y_MIN)) DEALLOCATE(SEG_Y_MIN, SEG_Y_MAX)
        IF (ALLOCATED(SEG_Z_MIN)) DEALLOCATE(SEG_Z_MIN, SEG_Z_MAX)
        IF (ALLOCATED(SLICE_BOX_ALL)) DEALLOCATE(SLICE_BOX_ALL)

    END SUBROUTINE PL_STATE_PROJECT_TO_GRID

    SUBROUTINE PL_STATE_BACKPROJECT_FROM_GRID(PL_STATE, PL_DS, BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
                        
        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(PL_DS_TYPE),       INTENT(IN)    :: PL_DS
        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS
        CLASS(BOXM_STATE_TYPE),  INTENT(IN)    :: BOXM_STATE
        CLASS(PATCH_STATE_TYPE), INTENT(IN)    :: PATCH_STATE
        INTEGER,                 INTENT(IN)    :: TIME_IDX

        INTEGER :: K, ROW_IDX, P, B, SEG_ID, CELL_C, CELL_F, PL_ID
        INTEGER :: NX_F, NY_F, I
        REAL(DP) :: DX_C_M, DY_C_M, DX_F, DY_F, VOL_F, LAT_C, TOTAL_PATCH_Y_DEL_F
        REAL(DP) :: SUM_W_REV
        INTEGER :: K2
        
        REAL(DP), ALLOCATABLE :: PL_MASS_NEW(:,:)
        LOGICAL, ALLOCATABLE :: SEG_HAS_MAP(:)
        
        IF (.NOT. ALLOCATED(PL_STATE%PL_MASS)) RETURN
        IF (.NOT. ALLOCATED(PL_STATE%SPECIES_PL_NUM)) RETURN
        IF (.NOT. ALLOCATED(PL_STATE%MAP_W)) RETURN
        IF (.NOT. ALLOCATED(PL_STATE%MAP_SEG)) RETURN
        IF (.NOT. ALLOCATED(PL_STATE%MAP_CELL_C)) RETURN
        IF (.NOT. ALLOCATED(PL_STATE%MAP_CELL_F)) RETURN
        IF (.NOT. ALLOCATED(PATCH_STATE%Y_DEL_F)) RETURN
        IF (.NOT. ALLOCATED(PATCH_STATE%ROW_CELL_C)) RETURN
        IF (.NOT. ALLOCATED(PATCH_STATE%ROW_CELL_F)) RETURN
        IF (.NOT. ALLOCATED(BOXM_STATE%MOL_MASS_C)) RETURN
        IF (.NOT. ALLOCATED(BOXM_STATE%LATITUDE_C)) RETURN
        IF (PL_STATE%NNZ_MAP <= 0) RETURN
        IF (PATCH_STATE%NROWS <= 0) RETURN

        IF (ALLOCATED(PL_STATE%PL_MASS)) THEN
            PRINT *, 'DEBUG: TOTAL PLUME MASS AT ENTRY (BACKPROJECTION) =', SUM(PL_STATE%PL_MASS)
        END IF
        CALL FLUSH(6)

        IF (ALLOCATED(PATCH_STATE%Y_DEL_F)) THEN
            TOTAL_PATCH_Y_DEL_F = 0.0_DP
            DO I = 1, SIZE(PATCH_STATE%Y_DEL_F, 1)
                TOTAL_PATCH_Y_DEL_F = TOTAL_PATCH_Y_DEL_F + PATCH_STATE%Y_DEL_F(I, 8)
            END DO
            PRINT *, 'DEBUG: TOTAL PATCH Y_DEL_F AT ENTRY (BACKPROJECTION) = ', TOTAL_PATCH_Y_DEL_F
            CALL FLUSH(6)
        END IF


        NX_F = MAX(1, NINT(BOXM_DS%HRES_SIM_C / BOXM_DS%HRES_SIM_F))
        NY_F = NX_F

        ! Start from previous plume mass so unmapped segments are preserved.
        IF (.NOT. ALLOCATED(PL_MASS_NEW)) ALLOCATE(PL_MASS_NEW(PL_STATE%NSEG, PL_STATE%NSPL))
        PL_MASS_NEW(:,:) = PL_STATE%PL_MASS(:,:)

        IF (.NOT. ALLOCATED(SEG_HAS_MAP)) ALLOCATE(SEG_HAS_MAP(PL_STATE%NSEG))
        SEG_HAS_MAP(:) = .FALSE.

        ! Mark which segments actually appear in the sparse map.
        DO K = 1, PL_STATE%NNZ_MAP
            SEG_ID = PL_STATE%MAP_SEG(K)
            IF (SEG_ID >= 1 .AND. SEG_ID <= PL_STATE%NSEG) THEN
                SEG_HAS_MAP(SEG_ID) = .TRUE.
            END IF
        END DO

        ! Only mapped segments get overwritten by backprojected values.
        DO SEG_ID = 1, PL_STATE%NSEG
            IF (SEG_HAS_MAP(SEG_ID)) THEN
                PL_MASS_NEW(SEG_ID,:) = 0.0_DP
            END IF
        END DO

        ! TIME_IDX and PL_DS are kept in signature for orchestration symmetry and future time-aware updates.
        IF (TIME_IDX < 0 .OR. PL_DS%NSEG < 0) RETURN

        DO K = 1, PL_STATE%NNZ_MAP
            SEG_ID = PL_STATE%MAP_SEG(K)
            CELL_C = PL_STATE%MAP_CELL_C(K)
            CELL_F = PL_STATE%MAP_CELL_F(K)

            IF (SEG_ID < 1 .OR. SEG_ID > PL_STATE%NSEG) CYCLE
            IF (CELL_C < 1 .OR. CELL_C > BOXM_STATE%NCELL) CYCLE

            ROW_IDX = -1
            DO P = 1, PATCH_STATE%NROWS
                IF (PATCH_STATE%ROW_CELL_C(P) == CELL_C .AND. PATCH_STATE%ROW_CELL_F(P) == CELL_F) THEN
                    ROW_IDX = P
                    EXIT
                END IF
            END DO
            IF (ROW_IDX < 1) CYCLE

            SUM_W_REV = 0.0_DP
            DO K2 = 1, PL_STATE%NNZ_MAP
                IF (PL_STATE%MAP_CELL_C(K2) == CELL_C .AND. PL_STATE%MAP_CELL_F(K2) == CELL_F) THEN
                    SUM_W_REV = SUM_W_REV + PL_STATE%MAP_W(K2)
                END IF
            END DO
            IF (SUM_W_REV <= 0.0_DP) CYCLE

            DX_C_M = BOXM_STATE%DX_C_M(CELL_C)
            DY_C_M = BOXM_STATE%DY_C_M(CELL_C)
            DX_F = DX_C_M / REAL(NX_F, DP)
            DY_F = DY_C_M / REAL(NY_F, DP)
            VOL_F = DX_F * DY_F * BOXM_DS%VRES_SIM_F
            IF (VOL_F <= 0.0_DP) CYCLE

            DO PL_ID = 1, PL_STATE%NSPL
                B = PL_STATE%SPECIES_PL_NUM(PL_ID)
                IF (B < 1 .OR. B > PATCH_STATE%NSBOXM) CYCLE
                IF (B > SIZE(BOXM_STATE%MOL_MASS_C)) CYCLE
                IF (PL_ID > SIZE(PL_STATE%PL_MASS, 2)) CYCLE
                IF (BOXM_STATE%MOL_MASS_C(B) /= BOXM_STATE%MOL_MASS_C(B)) CYCLE
                IF (BOXM_STATE%MOL_MASS_C(B) <= 0.0_DP) CYCLE

                ! IF (PL_ID .EQ. 1) THEN !  .OR. PL_ID .EQ. 13 .OR. PL_ID .EQ. 16
                !     PRINT *, 'DEBUG: K=', K, 'SEG_ID=', SEG_ID, 'CELL_C=', CELL_C, 'CELL_F=', CELL_F, 'PL_ID=', PL_ID, &
                !         'MAP_W=', PL_STATE%MAP_W(K), 'Y_DEL_F=', PATCH_STATE%Y_DEL_F(ROW_IDX, B), 'MOL_MASS_C=', &
                !         BOXM_STATE%MOL_MASS_C(B), 'VOL_F=', VOL_F
                !     FLUSH(6)
                ! END IF

                PL_MASS_NEW(SEG_ID, PL_ID) = PL_MASS_NEW(SEG_ID, PL_ID) + (PL_STATE%MAP_W(K) / SUM_W_REV) * &
                PATCH_STATE%Y_DEL_F(ROW_IDX, B) * BOXM_STATE%MOL_MASS_C(B) * VOL_F * 1.0E6_DP / 6.02214076E23_DP
            END DO
        END DO

        PL_STATE%PL_MASS(:,:) = PL_MASS_NEW(:,:)

        IF (ALLOCATED(PL_MASS_NEW)) DEALLOCATE(PL_MASS_NEW)
        IF (ALLOCATED(SEG_HAS_MAP)) DEALLOCATE(SEG_HAS_MAP)

        IF (ALLOCATED(PL_STATE%PL_MASS)) THEN
            PRINT *, 'DEBUG: TOTAL PLUME MASS AT EXIT (BACKPROJECTION) =', SUM(PL_STATE%PL_MASS)
        END IF
        CALL FLUSH(6)

        IF (ALLOCATED(PATCH_STATE%Y_DEL_F)) THEN
            TOTAL_PATCH_Y_DEL_F = 0.0_DP
            DO I = 1, SIZE(PATCH_STATE%Y_DEL_F, 1)
                TOTAL_PATCH_Y_DEL_F = TOTAL_PATCH_Y_DEL_F + PATCH_STATE%Y_DEL_F(I, 8)
            END DO
            PRINT *, 'DEBUG: TOTAL PATCH Y_DEL_F AT EXIT (BACKPROJECTION) = ', TOTAL_PATCH_Y_DEL_F
            CALL FLUSH(6)
        END IF
 
    END SUBROUTINE PL_STATE_BACKPROJECT_FROM_GRID

    ! ---------- BOXM STATE METHODS ----------
    SUBROUTINE BOXM_STATE_INIT_FROM_BOXM_DS(BOXM_STATE, BOXM_DS)
        CLASS(BOXM_STATE_TYPE), INTENT(INOUT) :: BOXM_STATE
        CLASS(BOXM_DS_TYPE),    INTENT(IN)    :: BOXM_DS

        INTEGER :: I

        BOXM_STATE%NCELL = BOXM_DS%NCELL
        BOXM_STATE%NSBOXM = BOXM_DS%NSBOXM

        ! ALLOCATE ARRAYS
        IF (.NOT. ALLOCATED(BOXM_STATE%SPECIES_BOXM_NUM)) ALLOCATE(BOXM_STATE%SPECIES_BOXM_NUM(BOXM_STATE%NSBOXM))
        IF (.NOT. ALLOCATED(BOXM_STATE%MOL_MASS_C)) ALLOCATE(BOXM_STATE%MOL_MASS_C(BOXM_STATE%NSBOXM))

        IF (.NOT. ALLOCATED(BOXM_STATE%LONGITUDE_C)) ALLOCATE(BOXM_STATE%LONGITUDE_C(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%LATITUDE_C)) ALLOCATE(BOXM_STATE%LATITUDE_C(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%ALTITUDE_C)) ALLOCATE(BOXM_STATE%ALTITUDE_C(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%LEVEL_C)) ALLOCATE(BOXM_STATE%LEVEL_C(BOXM_STATE%NCELL))

        IF (.NOT. ALLOCATED(BOXM_STATE%LONGITUDE_C_M)) ALLOCATE(BOXM_STATE%LONGITUDE_C_M(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%LATITUDE_C_M)) ALLOCATE(BOXM_STATE%LATITUDE_C_M(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%DX_C_M)) ALLOCATE(BOXM_STATE%DX_C_M(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%DY_C_M)) ALLOCATE(BOXM_STATE%DY_C_M(BOXM_STATE%NCELL))

        IF (.NOT. ALLOCATED(BOXM_STATE%TEMP)) ALLOCATE(BOXM_STATE%TEMP(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%H2O)) ALLOCATE(BOXM_STATE%H2O(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%M)) ALLOCATE(BOXM_STATE%M(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%O2)) ALLOCATE(BOXM_STATE%O2(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%N2)) ALLOCATE(BOXM_STATE%N2(BOXM_STATE%NCELL))
        IF (.NOT. ALLOCATED(BOXM_STATE%SZA)) ALLOCATE(BOXM_STATE%SZA(BOXM_STATE%NCELL))

        IF (.NOT. ALLOCATED(BOXM_STATE%Y_BG_C)) ALLOCATE(BOXM_STATE%Y_BG_C(BOXM_STATE%NCELL, BOXM_STATE%NSBOXM))
        IF (.NOT. ALLOCATED(BOXM_STATE%Y_DEL_C)) ALLOCATE(BOXM_STATE%Y_DEL_C(BOXM_STATE%NCELL, BOXM_STATE%NSBOXM))
        IF (.NOT. ALLOCATED(BOXM_STATE%ACTIVE_FLAG)) ALLOCATE(BOXM_STATE%ACTIVE_FLAG(BOXM_STATE%NCELL))

        BOXM_STATE%DTS = BOXM_DS%TS_SIM

        BOXM_STATE%TIME_REL_S = 0
        BOXM_STATE%SPECIES_BOXM_NUM(:) = BOXM_DS%SPECIES_BOXM_NUM(:)
        BOXM_STATE%MOL_MASS_C(:) = BOXM_DS%MOL_MASS_C(:)
        BOXM_STATE%LONGITUDE_C(:) = BOXM_DS%LONGITUDE_C(:)
        BOXM_STATE%LATITUDE_C(:) = BOXM_DS%LATITUDE_C(:)
        BOXM_STATE%ALTITUDE_C(:) = BOXM_DS%ALTITUDE_C(:)
        BOXM_STATE%LEVEL_C(:) = BOXM_DS%LEVEL_C(:)
        BOXM_STATE%LONGITUDE_C_M(:) = BOXM_DS%LONGITUDE_C_M(:)
        BOXM_STATE%LATITUDE_C_M(:) = BOXM_DS%LATITUDE_C_M(:)
        CALL BOXM_STATE_BUILD_CELL_METRICS(BOXM_STATE, BOXM_DS)
        BOXM_STATE%TEMP(:) = BOXM_DS%TEMP(:,1)
        BOXM_STATE%H2O(:) = BOXM_DS%H2O(:,1)
        BOXM_STATE%M(:) = BOXM_DS%M(:,1)
        BOXM_STATE%O2(:) = BOXM_DS%O2(:,1)
        BOXM_STATE%N2(:) = BOXM_DS%N2(:,1)
        BOXM_STATE%SZA(:) = BOXM_DS%SZA(:,1)
        BOXM_STATE%Y_BG_C(:,:) = BOXM_DS%Y_BG_C(:,:)
        BOXM_STATE%Y_DEL_C(:,:) = 0.0_DP
        BOXM_STATE%ACTIVE_FLAG(:) = .FALSE.

    END SUBROUTINE BOXM_STATE_INIT_FROM_BOXM_DS

    SUBROUTINE BOXM_STATE_BUILD_CELL_METRICS(BOXM_STATE, BOXM_DS)
        CLASS(BOXM_STATE_TYPE), INTENT(INOUT) :: BOXM_STATE
        CLASS(BOXM_DS_TYPE),    INTENT(IN)    :: BOXM_DS

        INTEGER :: I, J
        REAL(DP) :: LON_C, LAT_C, LEV_C, RES
        REAL(DP) :: DX_E, DX_W, DY_N, DY_S
        REAL(DP) :: LON_TOL, LAT_TOL, LEV_TOL
        LOGICAL :: HAVE_E, HAVE_W, HAVE_N, HAVE_S

        RES = BOXM_DS%HRES_SIM_C
        LON_TOL = MAX(1.0E-10_DP, 1.0E-3_DP * ABS(RES))
        LAT_TOL = LON_TOL
        LEV_TOL = 1.0E-6_DP

        DO I = 1, BOXM_STATE%NCELL
            LON_C = BOXM_STATE%LONGITUDE_C(I)
            LAT_C = BOXM_STATE%LATITUDE_C(I)
            LEV_C = BOXM_STATE%LEVEL_C(I)

            ! Fallback: spherical local metric if a neighbour lookup fails.
            BOXM_STATE%DX_C_M(I) = DEG_TO_M_DX(RES, LAT_C)
            BOXM_STATE%DY_C_M(I) = DEG_TO_M_DY(RES)

            HAVE_E = .FALSE.
            HAVE_W = .FALSE.
            HAVE_N = .FALSE.
            HAVE_S = .FALSE.
            DX_E = 0.0_DP
            DX_W = 0.0_DP
            DY_N = 0.0_DP
            DY_S = 0.0_DP

            DO J = 1, BOXM_STATE%NCELL
                IF (J == I) CYCLE
                IF (ABS(BOXM_STATE%LEVEL_C(J) - LEV_C) > LEV_TOL) CYCLE

                IF (ABS(BOXM_STATE%LATITUDE_C(J) - LAT_C) <= LAT_TOL) THEN
                    IF (ABS(BOXM_STATE%LONGITUDE_C(J) - (LON_C + RES)) <= LON_TOL) THEN
                        DX_E = ABS(BOXM_STATE%LONGITUDE_C_M(J) - BOXM_STATE%LONGITUDE_C_M(I))
                        HAVE_E = .TRUE.
                    ELSE IF (ABS(BOXM_STATE%LONGITUDE_C(J) - (LON_C - RES)) <= LON_TOL) THEN
                        DX_W = ABS(BOXM_STATE%LONGITUDE_C_M(I) - BOXM_STATE%LONGITUDE_C_M(J))
                        HAVE_W = .TRUE.
                    END IF
                END IF

                IF (ABS(BOXM_STATE%LONGITUDE_C(J) - LON_C) <= LON_TOL) THEN
                    IF (ABS(BOXM_STATE%LATITUDE_C(J) - (LAT_C + RES)) <= LAT_TOL) THEN
                        DY_N = ABS(BOXM_STATE%LATITUDE_C_M(J) - BOXM_STATE%LATITUDE_C_M(I))
                        HAVE_N = .TRUE.
                    ELSE IF (ABS(BOXM_STATE%LATITUDE_C(J) - (LAT_C - RES)) <= LAT_TOL) THEN
                        DY_S = ABS(BOXM_STATE%LATITUDE_C_M(I) - BOXM_STATE%LATITUDE_C_M(J))
                        HAVE_S = .TRUE.
                    END IF
                END IF
            END DO

            IF (HAVE_E .AND. HAVE_W) THEN
                BOXM_STATE%DX_C_M(I) = 0.5_DP * (DX_E + DX_W)
            ELSE IF (HAVE_E) THEN
                BOXM_STATE%DX_C_M(I) = DX_E
            ELSE IF (HAVE_W) THEN
                BOXM_STATE%DX_C_M(I) = DX_W
            END IF

            IF (HAVE_N .AND. HAVE_S) THEN
                BOXM_STATE%DY_C_M(I) = 0.5_DP * (DY_N + DY_S)
            ELSE IF (HAVE_N) THEN
                BOXM_STATE%DY_C_M(I) = DY_N
            ELSE IF (HAVE_S) THEN
                BOXM_STATE%DY_C_M(I) = DY_S
            END IF
        END DO
    END SUBROUTINE BOXM_STATE_BUILD_CELL_METRICS

    SUBROUTINE BOXM_STATE_ADVANCE_MET(BOXM_STATE, BOXM_DS, TIME_IDX)
        CLASS(BOXM_STATE_TYPE), INTENT(INOUT) :: BOXM_STATE
        CLASS(BOXM_DS_TYPE),    INTENT(IN)    :: BOXM_DS
        INTEGER, INTENT(IN) :: TIME_IDX

        BOXM_STATE%TIME_REL_S = BOXM_DS%TIME_REL_S(TIME_IDX)
        BOXM_STATE%TEMP(:) = BOXM_DS%TEMP(:,TIME_IDX)
        BOXM_STATE%H2O(:) = BOXM_DS%H2O(:,TIME_IDX)
        BOXM_STATE%M(:) = BOXM_DS%M(:,TIME_IDX)
        BOXM_STATE%O2(:) = BOXM_DS%O2(:,TIME_IDX)
        BOXM_STATE%N2(:) = BOXM_DS%N2(:,TIME_IDX)
        BOXM_STATE%SZA(:) = BOXM_DS%SZA(:,TIME_IDX)

    END SUBROUTINE BOXM_STATE_ADVANCE_MET

    SUBROUTINE BOXM_STATE_RUN_COARSE_BG_CHEM(BOXM_STATE, BOXM_DS)
        CLASS(BOXM_STATE_TYPE), INTENT(INOUT) :: BOXM_STATE
        CLASS(BOXM_DS_TYPE),    INTENT(IN)    :: BOXM_DS

        CALL RUN_LEGACY_CHEM(BOXM_STATE%Y_BG_C, BOXM_STATE%TEMP, BOXM_STATE%H2O, BOXM_STATE%O2, &
                            BOXM_STATE%N2, BOXM_STATE%M, BOXM_STATE%SZA, BOXM_STATE%DTS, BOXM_STATE%NCELL, &
                            BOXM_DS%NSBOXM, BOXM_DS%NPP, BOXM_DS%NPC, BOXM_DS%NTC, BOXM_DS%NFL)

    END SUBROUTINE BOXM_STATE_RUN_COARSE_BG_CHEM

    SUBROUTINE BOXM_STATE_RUN_COARSE_DELTA_CHEM(BOXM_STATE, BOXM_DS, PATCH_STATE)
        CLASS(BOXM_STATE_TYPE), INTENT(INOUT) :: BOXM_STATE
        CLASS(BOXM_DS_TYPE),    INTENT(IN)    :: BOXM_DS
        CLASS(PATCH_STATE_TYPE), INTENT(IN)   :: PATCH_STATE

        INTEGER :: CELL_ID, SPECIES_ID, ROW_IDX
        REAL(DP) :: TOTAL_MASS, COARSE_VOL, FINE_VOL

        ! Mass-weighted averaging: sum fine cell mass, divide by coarse cell volume
        BOXM_STATE%Y_DEL_C(:,:) = 0.0_DP
        
        DO CELL_ID = 1, BOXM_STATE%NCELL
            COARSE_VOL = BOXM_STATE%DX_C_M(CELL_ID) * BOXM_STATE%DY_C_M(CELL_ID) * BOXM_DS%VRES_SIM_C
            IF (COARSE_VOL <= 0.0_DP) CYCLE
            DO SPECIES_ID = 1, BOXM_STATE%NSBOXM
                TOTAL_MASS = 0.0_DP
                DO ROW_IDX = 1, PATCH_STATE%NROWS
                    IF (PATCH_STATE%ROW_CELL_C(ROW_IDX) == CELL_ID) THEN
                        FINE_VOL = (BOXM_STATE%DX_C_M(CELL_ID) / MAX(1, NINT(BOXM_DS%HRES_SIM_C / BOXM_DS%HRES_SIM_F))) * &
                                   (BOXM_STATE%DY_C_M(CELL_ID) / MAX(1, NINT(BOXM_DS%HRES_SIM_C / BOXM_DS%HRES_SIM_F))) * &
                                    BOXM_DS%VRES_SIM_F
                        TOTAL_MASS = TOTAL_MASS + PATCH_STATE%Y_DEL_F(ROW_IDX, SPECIES_ID) * FINE_VOL
                    END IF
                END DO
                BOXM_STATE%Y_DEL_C(CELL_ID, SPECIES_ID) = TOTAL_MASS / COARSE_VOL
            END DO
        END DO

        CALL RUN_LEGACY_CHEM(BOXM_STATE%Y_DEL_C, BOXM_STATE%TEMP, BOXM_STATE%H2O, BOXM_STATE%O2, &
                            BOXM_STATE%N2, BOXM_STATE%M, BOXM_STATE%SZA, BOXM_STATE%DTS, BOXM_STATE%NCELL, &
                            BOXM_DS%NSBOXM, BOXM_DS%NPP, BOXM_DS%NPC, BOXM_DS%NTC, BOXM_DS%NFL)

    END SUBROUTINE BOXM_STATE_RUN_COARSE_DELTA_CHEM

    ! ---------- PATCH STATE METHODS ----------
    SUBROUTINE PATCH_STATE_INIT_FROM_BOXM_DS(PATCH_STATE, BOXM_DS)
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE
        CLASS(BOXM_DS_TYPE),    INTENT(IN)    :: BOXM_DS
        PATCH_STATE%NSBOXM = BOXM_DS%NSBOXM
        IF (.NOT. ALLOCATED(PATCH_STATE%SPECIES_BOXM_NUM)) THEN
            ALLOCATE(PATCH_STATE%SPECIES_BOXM_NUM(PATCH_STATE%NSBOXM))
        END IF
        PATCH_STATE%SPECIES_BOXM_NUM(:) = BOXM_DS%SPECIES_BOXM_NUM(:)

    END SUBROUTINE PATCH_STATE_INIT_FROM_BOXM_DS

    SUBROUTINE PATCH_STATE_BUILD_ROWS_FROM_W(PATCH_STATE, PL_STATE, BOXM_STATE)
        IMPLICIT NONE
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE
        CLASS(PL_STATE_TYPE),    INTENT(IN)    :: PL_STATE
        CLASS(BOXM_STATE_TYPE),  INTENT(IN)    :: BOXM_STATE

        INTEGER :: I, J, ROW_IDX, CELL_C, CELL_F
        INTEGER :: CELL_C_CAND, CELL_F_CAND
        INTEGER :: NNZ
        LOGICAL :: FOUND
        INTEGER, ALLOCATABLE :: TMP_CELL_C(:), TMP_CELL_F(:)

        PATCH_STATE%NSBOXM = BOXM_STATE%NSBOXM
        PATCH_STATE%NROWS = 0
        PATCH_STATE%ROW_IDX = 0

        NNZ = PL_STATE%NNZ_MAP
        IF (NNZ <= 0) RETURN

        IF (ALLOCATED(PATCH_STATE%TIME_REL_S)) DEALLOCATE(PATCH_STATE%TIME_REL_S)
        IF (ALLOCATED(PATCH_STATE%ROW_CELL_C)) DEALLOCATE(PATCH_STATE%ROW_CELL_C)
        IF (ALLOCATED(PATCH_STATE%ROW_CELL_F)) DEALLOCATE(PATCH_STATE%ROW_CELL_F)
        IF (ALLOCATED(PATCH_STATE%Y_DEL_F)) DEALLOCATE(PATCH_STATE%Y_DEL_F)

        ALLOCATE(TMP_CELL_C(NNZ), TMP_CELL_F(NNZ))
        TMP_CELL_C(:) = 0
        TMP_CELL_F(:) = 0

        ! Build one PATCH row per unique (coarse cell, fine subcell) pair found in W.
        DO I = 1, NNZ
            FOUND = .FALSE.
            DO J = 1, PATCH_STATE%NROWS
                IF (TMP_CELL_C(J) == PL_STATE%MAP_CELL_C(I) .AND. TMP_CELL_F(J) == PL_STATE%MAP_CELL_F(I)) THEN
                    FOUND = .TRUE.
                    EXIT
                END IF
            END DO
            IF (.NOT. FOUND) THEN
                PATCH_STATE%NROWS = PATCH_STATE%NROWS + 1
                TMP_CELL_C(PATCH_STATE%NROWS) = PL_STATE%MAP_CELL_C(I)
                TMP_CELL_F(PATCH_STATE%NROWS) = PL_STATE%MAP_CELL_F(I)
            END IF
        END DO

        IF (PATCH_STATE%NROWS <= 0) THEN
            DEALLOCATE(TMP_CELL_C, TMP_CELL_F)
            RETURN
        END IF

        ALLOCATE(PATCH_STATE%TIME_REL_S(PATCH_STATE%NROWS))
        ALLOCATE(PATCH_STATE%ROW_CELL_C(PATCH_STATE%NROWS))
        ALLOCATE(PATCH_STATE%ROW_CELL_F(PATCH_STATE%NROWS))
        ALLOCATE(PATCH_STATE%Y_DEL_F(PATCH_STATE%NROWS, PATCH_STATE%NSBOXM))

        DO ROW_IDX = 1, PATCH_STATE%NROWS
            CELL_C = TMP_CELL_C(ROW_IDX)
            CELL_F = TMP_CELL_F(ROW_IDX)

            PATCH_STATE%TIME_REL_S(ROW_IDX) = BOXM_STATE%TIME_REL_S
            PATCH_STATE%ROW_CELL_C(ROW_IDX) = CELL_C
            PATCH_STATE%ROW_CELL_F(ROW_IDX) = CELL_F
        END DO
        PATCH_STATE%ROW_IDX = PATCH_STATE%NROWS

        DEALLOCATE(TMP_CELL_C, TMP_CELL_F)

    END SUBROUTINE PATCH_STATE_BUILD_ROWS_FROM_W

    SUBROUTINE PATCH_STATE_ACCUM_DELTAS_FROM_W(PATCH_STATE, PL_STATE, BOXM_DS, BOXM_STATE)
        IMPLICIT NONE
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE
        CLASS(PL_STATE_TYPE),    INTENT(IN)    :: PL_STATE
        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS
        CLASS(BOXM_STATE_TYPE),  INTENT(IN)    :: BOXM_STATE

        INTEGER  :: K, ROW_IDX, PL_ID, BOXM_ID, SEG_ID, SLICE_ID, CELL_C
        INTEGER  :: NX_F, NY_F
        REAL(DP) :: W, VOL_F
        REAL(DP) :: DX_C_M, DY_C_M, DX_F, DY_F
        INTEGER :: ROW
        
        IF (PATCH_STATE%NROWS <= 0 .OR. PL_STATE%NNZ_MAP <= 0) RETURN

        ! Compute fine-grid subdivision factors once.
        NX_F = MAX(1, NINT(BOXM_DS%HRES_SIM_C / BOXM_DS%HRES_SIM_F))
        NY_F = NX_F

        PATCH_STATE%Y_DEL_F(:,:) = 0.0_DP

        DO K = 1, PL_STATE%NNZ_MAP
            SEG_ID = PL_STATE%MAP_SEG(K)
            SLICE_ID = PL_STATE%MAP_SLICE(K)
            W      = PL_STATE%MAP_W(K)
            CELL_C = PL_STATE%MAP_CELL_C(K)

            IF (CELL_C < 1 .OR. CELL_C > BOXM_STATE%NCELL) CYCLE

            DX_C_M = BOXM_STATE%DX_C_M(CELL_C)
            DY_C_M = BOXM_STATE%DY_C_M(CELL_C)
            DX_F = DX_C_M / REAL(NX_F, DP)
            DY_F = DY_C_M / REAL(NY_F, DP)
            VOL_F = DX_F * DY_F * BOXM_DS%VRES_SIM_F
            IF (VOL_F <= 0.0_DP) CYCLE

            ! Find patch row for this (cell_c, cell_f)
            ROW_IDX = -1
            DO ROW_IDX = 1, PATCH_STATE%NROWS
                IF (PATCH_STATE%ROW_CELL_C(ROW_IDX) == PL_STATE%MAP_CELL_C(K) .AND. &
                    PATCH_STATE%ROW_CELL_F(ROW_IDX) == PL_STATE%MAP_CELL_F(K)) EXIT
            END DO
            IF (ROW_IDX > PATCH_STATE%NROWS) CYCLE   ! not found — should not happen

            ! Accumulate plume species into boxm species slots
            DO PL_ID = 1, PL_STATE%NSPL
                BOXM_ID = PL_STATE%SPECIES_PL_NUM(PL_ID)
                IF (BOXM_ID < 1 .OR. BOXM_ID > PATCH_STATE%NSBOXM) CYCLE
                IF (BOXM_STATE%MOL_MASS_C(BOXM_ID) <= 0.0_DP) CYCLE
                PATCH_STATE%Y_DEL_F(ROW_IDX, BOXM_ID) = PATCH_STATE%Y_DEL_F(ROW_IDX, BOXM_ID) + &
                    W * (PL_STATE%PL_MASS(SEG_ID, PL_ID) * 6.02214076E23_DP / &
                        (BOXM_STATE%MOL_MASS_C(BOXM_ID) * VOL_F * 1.0E6_DP)) 
            END DO
        END DO

    END SUBROUTINE PATCH_STATE_ACCUM_DELTAS_FROM_W

    SUBROUTINE PATCH_STATE_RUN_FINE_DELTA_CHEM(PATCH_STATE, BOXM_DS, BOXM_STATE)
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE
        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS
        CLASS(BOXM_STATE_TYPE),  INTENT(IN)    :: BOXM_STATE

        INTEGER :: CELL_C, I, J, ROW_IDX
        INTEGER :: NX_F, NY_F, NFINE_XY, NFINE_Z, NFINE_CELL
        INTEGER, ALLOCATABLE :: ROWS_CELL(:)
        REAL(DP), ALLOCATABLE :: Y_BG_F_CELL(:,:), Y_DEL_F_CELL(:,:), Y_TOT_F_CELL(:,:)

        REAL(DP), ALLOCATABLE :: TEMP_CELL(:), H2O_CELL(:), O2_CELL(:), N2_CELL(:), M_CELL(:), SZA_CELL(:)
        LOGICAL :: FOUND

        ! Compute fine-grid subdivision factors once.

        NFINE_XY = MAX(1, NINT(BOXM_DS%HRES_SIM_C / BOXM_DS%HRES_SIM_F))
        NFINE_Z = MAX(1, NINT(BOXM_DS%VRES_SIM_C / BOXM_DS%VRES_SIM_F))
        NFINE_CELL = NFINE_XY * NFINE_XY * NFINE_Z

        ! Placeholder until row-wise chemistry update is implemented.
        IF (PATCH_STATE%NROWS <= 0) RETURN

        DO CELL_C = 1, BOXM_STATE%NCELL
            IF (.NOT. BOXM_STATE%ACTIVE_FLAG(CELL_C)) CYCLE

            ! FIND ALL RELEVANT ROWS TO THIS COARSE CELL
            ROWS_CELL = PACK((/(I, I=1,PATCH_STATE%NROWS)/), PATCH_STATE%ROW_CELL_C(:) == CELL_C)

            ! ALLOCATE MET VARIABLES
            ALLOCATE(TEMP_CELL(NFINE_CELL), H2O_CELL(NFINE_CELL), O2_CELL(NFINE_CELL), &
                     N2_CELL(NFINE_CELL), M_CELL(NFINE_CELL), SZA_CELL(NFINE_CELL))

            ! ALLOCATE Y (CONCENTRATION) VARIABLES
            ALLOCATE(Y_BG_F_CELL(NFINE_CELL, BOXM_STATE%NSBOXM))
            ALLOCATE(Y_DEL_F_CELL(NFINE_CELL, PATCH_STATE%NSBOXM))
            ALLOCATE(Y_TOT_F_CELL(NFINE_CELL, BOXM_STATE%NSBOXM))

            Y_BG_F_CELL(:,:) = 0.0_DP
            Y_DEL_F_CELL(:,:) = 0.0_DP
            Y_TOT_F_CELL(:,:) = 0.0_DP

            ! LOCATE MET VALUES FROM BOXM STATE
            TEMP_CELL(:) = BOXM_STATE%TEMP(CELL_C)
            H2O_CELL(:) = BOXM_STATE%H2O(CELL_C)
            O2_CELL(:) = BOXM_STATE%O2(CELL_C)
            N2_CELL(:) = BOXM_STATE%N2(CELL_C)
            M_CELL(:) = BOXM_STATE%M(CELL_C)
            SZA_CELL(:) = BOXM_STATE%SZA(CELL_C)

            ! LOCATE Y_BG_F VALUES FROM BOXM STATE
            DO I = 1, NFINE_CELL
                Y_BG_F_CELL(I,:) = BOXM_STATE%Y_BG_C(CELL_C,:)
            END DO

            ! LOCATE Y_DEL_F VALUES FROM PATCH STATE
            DO I = 1, NFINE_CELL
                FOUND = .FALSE.
                DO J = 1, SIZE(ROWS_CELL)
                    ROW_IDX = ROWS_CELL(J)
                    IF (PATCH_STATE%ROW_CELL_F(ROW_IDX) == I) THEN
                        Y_DEL_F_CELL(I,:) = PATCH_STATE%Y_DEL_F(ROW_IDX,:)
                        FOUND = .TRUE.
                        EXIT
                    END IF
                END DO
                ! If not found, Y_DEL_F_CELL(I,:) remains zero
            END DO
            
            ! SUM TOTAL MASS INTO CELLS FOR CHEMISTRY
            Y_TOT_F_CELL = Y_BG_F_CELL + Y_DEL_F_CELL

            ! ACCESS Y_TOT_F_CELL(I, :) FOR CHEMISTRY UPDATE
            CALL RUN_LEGACY_CHEM(Y_TOT_F_CELL, TEMP_CELL, H2O_CELL, O2_CELL, N2_CELL, M_CELL, &
                            SZA_CELL, BOXM_STATE%DTS, NFINE_CELL, BOXM_DS%NSBOXM, BOXM_DS%NPP, &
                            BOXM_DS%NPC, BOXM_DS%NTC, BOXM_DS%NFL)

            ! SUBTRACT UPDATED BG CHEM FROM TOTAL TO GET REMAINING IN PLUME
            Y_DEL_F_CELL = Y_TOT_F_CELL - Y_BG_F_CELL

            ! WRITE BACK RESULTS TO PATCH_STATE%Y_DEL_F
            DO I = 1, NFINE_CELL
                FOUND = .FALSE.
                DO J = 1, SIZE(ROWS_CELL)
                    ROW_IDX = ROWS_CELL(J)
                    IF (PATCH_STATE%ROW_CELL_F(ROW_IDX) == I) THEN
                        PATCH_STATE%Y_DEL_F(ROW_IDX,:) = Y_DEL_F_CELL(I,:)
                        FOUND = .TRUE.
                        EXIT
                    END IF
                END DO
                ! If not found, this would be unexpected since we are iterating over rows for this cell, 
                ! but we can choose to ignore or log as needed.
            END DO
            DEALLOCATE(Y_BG_F_CELL)
            DEALLOCATE(Y_DEL_F_CELL)
            DEALLOCATE(Y_TOT_F_CELL)
            DEALLOCATE(ROWS_CELL)
            DEALLOCATE(TEMP_CELL, H2O_CELL, O2_CELL, N2_CELL, M_CELL, SZA_CELL)
        END DO
    END SUBROUTINE PATCH_STATE_RUN_FINE_DELTA_CHEM

END MODULE DEFINE_STATE_TYPES

MODULE DEFINE_OUTPUT_TYPES

    USE NETCDF
    USE HELPERS
    USE DEFINE_INPUT_TYPES
    USE DEFINE_STATE_TYPES
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: PL_OUT_TYPE, BOXM_OUT_TYPE, PATCH_TABLE_TYPE

    TYPE(PL_DS_TYPE) :: PL_DS

    ! ------- OUTPUT DATA TYPES -------
    TYPE :: PL_OUT_TYPE
        
        ! PL OUT DIMLENS
        INTEGER :: NSEG = 0
        INTEGER :: NSPL = 0
        INTEGER :: NTOUT = 0
        
        ! PL OUT COORDS
        INTEGER, ALLOCATABLE :: TIME_REL_S(:)
        INTEGER, ALLOCATABLE :: TIME_IDX(:)
        INTEGER, ALLOCATABLE :: SPECIES_PL_NUM(:)

        ! PL OUT VARS
        REAL(DP), ALLOCATABLE :: PL_MASS(:,:,:) ! [NSEG, NSPL, NTOUT]

        ! PRIVATE NETCDF PLUMBING
        INTEGER, PRIVATE :: PL_OUT_NCID = -1

        ! PL_OUT DIM IDs
        INTEGER, PRIVATE :: DIMID_SEG_ID = -1
        INTEGER, PRIVATE :: DIMID_TIME = -1
        INTEGER, PRIVATE :: DIMID_SPECIES_PL = -1

        ! PL_OUT VAR IDs
        INTEGER, PRIVATE :: VARID_TIME_REL_S = -1
        INTEGER, PRIVATE :: VARID_TIME_IDX = -1
        INTEGER, PRIVATE :: VARID_SPECIES_PL_NUM = -1
        INTEGER, PRIVATE :: VARID_PL_MASS = -1

        INTEGER, PRIVATE :: VARID_W_SLICE = -1
        INTEGER, PRIVATE :: VARID_Y_HALF = -1
        INTEGER, PRIVATE :: VARID_Z_HALF = -1
        INTEGER, PRIVATE :: VARID_M_FRAC = -1
        INTEGER, PRIVATE :: VARID_ELLIPSES_M = -1
        INTEGER, PRIVATE :: VARID_SLICE_POLYS_M = -1

        ! PL_SLICES DIMLENS
        INTEGER :: NSLICES = 0

        ! PL_SLICES VARS
        REAL(DP), ALLOCATABLE :: Y_HALF(:,:,:) ! [NSEG, NSPL, NTOUT]
        REAL(DP), ALLOCATABLE :: Z_HALF(:,:,:) ! [NSEG, NSPL, NTOUT]
        REAL(DP), ALLOCATABLE :: M_FRAC(:)      ! [NSEG]
        REAL(DP), ALLOCATABLE :: W_SLICE(:)     ! [NSPL]

        REAL(DP), ALLOCATABLE :: ELLIPSES_M(:,:,:,:) ! [NSEG, NPOINTS, 3, NTOUT]
        REAL(DP), ALLOCATABLE :: SLICE_POLYS_M(:,:,:,:,:) ! [NSEG, NSLICES, 4, 3, NTOUT]

        ! NETCDF STATUS
        LOGICAL, PRIVATE :: IS_OPEN = .FALSE.

    CONTAINS
        PROCEDURE, PASS :: INIT => PL_OUT_INIT
        PROCEDURE, PASS :: READ_STATIC => PL_OUT_READ_STATIC
        PROCEDURE, PASS :: WRITE => PL_OUT_WRITE
        PROCEDURE, PASS :: CLOSE => PL_OUT_CLOSE

    END TYPE PL_OUT_TYPE

    TYPE :: BOXM_OUT_TYPE
        ! BOXM OUT DIMLENS
        INTEGER :: NCELL = 0
        INTEGER :: NSOUT = 219
        INTEGER :: NTOUT = 0

        ! BOXM OUT COORDS
        INTEGER, ALLOCATABLE :: TIME_REL_S(:)
        INTEGER, ALLOCATABLE :: TIME_IDX(:)
        INTEGER, ALLOCATABLE :: SPECIES_OUT_NUM(:)

        ! BOXM OUT VARS
        REAL(DP), ALLOCATABLE :: Y_BG_C(:,:,:)
        REAL(DP), ALLOCATABLE :: Y_DEL_C(:,:,:)
        LOGICAL, ALLOCATABLE :: ACTIVE_FLAG(:,:)

        ! PRIVATE NETCDF PLUMBING
        INTEGER, PRIVATE :: BOXM_OUT_NCID = -1

        ! BOXM_OUT DIM IDs
        INTEGER, PRIVATE :: DIMID_TIME = -1
        INTEGER, PRIVATE :: DIMID_CELL = -1
        INTEGER, PRIVATE :: DIMID_SPECIES_OUT = -1

        ! BOXM_OUT VAR IDs
        INTEGER, PRIVATE :: VARID_TIME_REL_S = -1
        INTEGER, PRIVATE :: VARID_TIME_IDX = -1
        INTEGER, PRIVATE :: VARID_SPECIES_OUT_NUM = -1
        
        INTEGER, PRIVATE :: VARID_Y_BG_C = -1
        INTEGER, PRIVATE :: VARID_Y_DEL_C = -1
        INTEGER, PRIVATE :: VARID_ACTIVE_FLAG = -1


        ! NETCDF STATUS
        LOGICAL, PRIVATE :: IS_OPEN = .FALSE.
    CONTAINS
        PROCEDURE, PASS :: INIT => BOXM_OUT_INIT
        PROCEDURE, PASS :: READ_STATIC => BOXM_OUT_READ_STATIC
        PROCEDURE, PASS :: WRITE => BOXM_OUT_WRITE
        PROCEDURE, PASS :: CLOSE => BOXM_OUT_CLOSE
    
    END TYPE BOXM_OUT_TYPE

    TYPE :: PATCH_TABLE_TYPE

        ! PATCH TABLE DIMLENS
        INTEGER :: NROWS = 0
        INTEGER :: NSOUT = 0
        INTEGER :: NEXT_ROW = 1

        ! PATCH TABLE COORDS

        INTEGER, ALLOCATABLE :: TIME_REL_S(:)
        INTEGER, ALLOCATABLE :: TIME_IDX(:)
        INTEGER, ALLOCATABLE :: SPECIES_OUT_NUM(:)

        INTEGER, ALLOCATABLE :: ROW_CELL_C(:)
        INTEGER, ALLOCATABLE :: ROW_CELL_F(:)

        REAL(DP), ALLOCATABLE :: LATITUDE_F(:)
        REAL(DP), ALLOCATABLE :: LONGITUDE_F(:)
        REAL(DP), ALLOCATABLE :: ALTITUDE_F(:)
        REAL(DP), ALLOCATABLE :: LEVEL_F(:)
        
        ! PATCH TABLE VARS
        REAL(DP), ALLOCATABLE :: Y_DEL_F(:,:)

        ! PRIVATE NETCDF PLUMBING
        INTEGER, PRIVATE :: PATCH_TABLE_NCID = -1

        ! PATCH TABLE DIM IDs
        INTEGER, PRIVATE :: DIMID_ROW = -1
        INTEGER, PRIVATE :: DIMID_SPECIES_OUT = -1

        ! PATCH TABLE VAR IDs
        INTEGER, PRIVATE :: VARID_ROW = -1
        INTEGER, PRIVATE :: VARID_SPECIES_OUT_NUM = -1
        INTEGER, PRIVATE :: VARID_ROW_CELL_C = -1
        INTEGER, PRIVATE :: VARID_ROW_CELL_F = -1
        INTEGER, PRIVATE :: VARID_TIME_REL_S = -1
        INTEGER, PRIVATE :: VARID_TIME_IDX = -1
        INTEGER, PRIVATE :: VARID_LATITUDE_F = -1
        INTEGER, PRIVATE :: VARID_LONGITUDE_F = -1
        INTEGER, PRIVATE :: VARID_ALTITUDE_F = -1
        INTEGER, PRIVATE :: VARID_LEVEL_F = -1

        INTEGER, PRIVATE :: VARID_Y_DEL_F = -1
        
        ! NETCDF STATUS
        LOGICAL, PRIVATE :: IS_OPEN = .FALSE.

    CONTAINS
        PROCEDURE, PASS :: INIT => PATCH_TABLE_INIT
        PROCEDURE, PASS :: READ_STATIC => PATCH_TABLE_READ_STATIC
        PROCEDURE, PASS :: WRITE => PATCH_TABLE_WRITE
        PROCEDURE, PASS :: CLOSE => PATCH_TABLE_CLOSE

    END TYPE PATCH_TABLE_TYPE

CONTAINS
    ! ... PL_OUT METHODS ...
    SUBROUTINE PL_OUT_INIT(PL_OUT, PL_DS, FILEPATH)
        CLASS(PL_OUT_TYPE), INTENT(INOUT) :: PL_OUT
        CLASS(PL_DS_TYPE),  INTENT(IN)    :: PL_DS
        CHARACTER(LEN=*),  INTENT(IN)    :: FILEPATH

        INTEGER  :: STATUS

        IF (PL_OUT%IS_OPEN) THEN
            STATUS = NF90_CLOSE(PL_OUT%PL_OUT_NCID)
            CALL NC_CHECK(STATUS, "NF90_CLOSE(PL_OUT) IN PL_OUT_INIT")
        END IF

        STATUS = NF90_OPEN(TRIM(FILEPATH), NF90_WRITE, PL_OUT%PL_OUT_NCID)
        CALL NC_CHECK(STATUS, "NF90_OPEN(PL_OUT)")

        PL_OUT%IS_OPEN = .TRUE.

        ! DIM IDS AND LENGTHS
        STATUS = NF90_INQ_DIMID(PL_OUT%PL_OUT_NCID, "seg_id", PL_OUT%DIMID_SEG_ID)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(seg_id)")

        STATUS = NF90_INQUIRE_DIMENSION(PL_OUT%PL_OUT_NCID, PL_OUT%DIMID_SEG_ID, LEN=PL_OUT%NSEG)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(seg_id)")

        STATUS = NF90_INQ_DIMID(PL_OUT%PL_OUT_NCID, "species_pl", PL_OUT%DIMID_SPECIES_PL)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(species_out)")

        STATUS = NF90_INQUIRE_DIMENSION(PL_OUT%PL_OUT_NCID, PL_OUT%DIMID_SPECIES_PL, LEN=PL_OUT%NSPL)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(species_pl)")

        STATUS = NF90_INQ_DIMID(PL_OUT%PL_OUT_NCID, "time", PL_OUT%DIMID_TIME)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(time)")

        STATUS = NF90_INQUIRE_DIMENSION(PL_OUT%PL_OUT_NCID, PL_OUT%DIMID_TIME, LEN=PL_OUT%NTOUT)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(time)")

        ! VAR IDS
        STATUS = NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "time_rel_s", PL_OUT%VARID_TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_rel_s)")

        STATUS= NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "time_idx", PL_OUT%VARID_TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_idx)")

        STATUS = NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "species_pl_num", PL_OUT%VARID_SPECIES_PL_NUM)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(species_out_num)")

        STATUS = NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "pl_mass", PL_OUT%VARID_PL_MASS)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(pl_mass)")

        IF (PL_DS%OUTPUT_PL_SLICES == 1) THEN
            
            STATUS = NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "y_half", PL_OUT%VARID_Y_HALF)
            CALL NC_CHECK(STATUS, "NF90_INQ_VARID(y_half)")

            STATUS = NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "z_half", PL_OUT%VARID_Z_HALF)
            CALL NC_CHECK(STATUS, "NF90_INQ_VARID(z_half)")

            STATUS = NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "m_frac", PL_OUT%VARID_M_FRAC)
            CALL NC_CHECK(STATUS, "NF90_INQ_VARID(m_frac)")

            STATUS = NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "w_slice", PL_OUT%VARID_W_SLICE)
            CALL NC_CHECK(STATUS, "NF90_INQ_VARID(w_slice)")

            STATUS = NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "ellipses_m", PL_OUT%VARID_ELLIPSES_M)
            CALL NC_CHECK(STATUS, "NF90_INQ_VARID(ellipses_m)")

            STATUS = NF90_INQ_VARID(PL_OUT%PL_OUT_NCID, "slice_polys_m", PL_OUT%VARID_SLICE_POLYS_M)
            CALL NC_CHECK(STATUS, "NF90_INQ_VARID(slice_polys_m)")

            PL_OUT%NSLICES = PL_DS%NSLICES
        END IF

    END SUBROUTINE PL_OUT_INIT

    SUBROUTINE PL_OUT_READ_STATIC(PL_OUT, PL_DS)
        CLASS(PL_OUT_TYPE), INTENT(INOUT) :: PL_OUT
        CLASS(PL_DS_TYPE),    INTENT(IN)  :: PL_DS

        INTEGER :: STATUS

        IF (.NOT. PL_OUT%IS_OPEN) STOP "PL_OUT_READ_STATIC: FILE NOT OPEN (CALL INIT FIRST)"
        IF (PL_OUT%NSEG <= 0) STOP "PL_OUT_READ_STATIC: NSEG NOT SET"
        IF (PL_OUT%NSPL <= 0) STOP "PL_OUT_READ_STATIC: NSPL NOT SET"
        IF (PL_OUT%NTOUT <= 0) STOP "PL_OUT_READ_STATIC: NTOUT NOT SET"
        IF (PL_DS%OUTPUT_PL_SLICES == 1 .AND. PL_OUT%NSLICES <= 0) STOP "PL_OUT_READ_STATIC: NSLICES NOT SET"

        IF (.NOT. ALLOCATED(PL_OUT%TIME_REL_S)) ALLOCATE(PL_OUT%TIME_REL_S(PL_OUT%NTOUT))
        IF (.NOT. ALLOCATED(PL_OUT%TIME_IDX)) ALLOCATE(PL_OUT%TIME_IDX(PL_OUT%NTOUT))
        IF (.NOT. ALLOCATED(PL_OUT%SPECIES_PL_NUM)) ALLOCATE(PL_OUT%SPECIES_PL_NUM(PL_OUT%NSPL))
        IF (.NOT. ALLOCATED(PL_OUT%PL_MASS)) ALLOCATE(PL_OUT%PL_MASS(PL_OUT%NSEG, PL_OUT%NSPL, PL_OUT%NTOUT))

        IF (PL_DS%OUTPUT_PL_SLICES == 1) THEN
            IF (.NOT. ALLOCATED(PL_OUT%Y_HALF)) ALLOCATE(PL_OUT%Y_HALF(PL_OUT%NSEG, PL_OUT%NSLICES, PL_OUT%NTOUT))
            IF (.NOT. ALLOCATED(PL_OUT%Z_HALF)) ALLOCATE(PL_OUT%Z_HALF(PL_OUT%NSEG, PL_OUT%NSLICES, PL_OUT%NTOUT))
            IF (.NOT. ALLOCATED(PL_OUT%M_FRAC)) ALLOCATE(PL_OUT%M_FRAC(PL_OUT%NSLICES))
            IF (.NOT. ALLOCATED(PL_OUT%W_SLICE)) ALLOCATE(PL_OUT%W_SLICE(PL_OUT%NSLICES))
            IF (.NOT. ALLOCATED(PL_OUT%ELLIPSES_M)) ALLOCATE(PL_OUT%ELLIPSES_M(PL_OUT%NSEG, PL_DS%NPOINTS, 3, PL_OUT%NTOUT))
            IF (.NOT. ALLOCATED(PL_OUT%SLICE_POLYS_M)) ALLOCATE(PL_OUT%SLICE_POLYS_M(PL_OUT%NSEG, &
                                                                PL_OUT%NSLICES, 4, 3, PL_OUT%NTOUT))
        END IF

        STATUS = NF90_GET_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_TIME_REL_S, PL_OUT%TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_rel_s)")

        STATUS = NF90_GET_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_TIME_IDX, PL_OUT%TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_idx)")

        STATUS = NF90_GET_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_SPECIES_PL_NUM, PL_OUT%SPECIES_PL_NUM)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(species_pl_num)")

    END SUBROUTINE PL_OUT_READ_STATIC

    SUBROUTINE PL_OUT_WRITE(PL_OUT, PL_STATE, PL_DS, TIME_IDX)
        CLASS(PL_OUT_TYPE), INTENT(INOUT) :: PL_OUT
        CLASS(PL_STATE_TYPE), INTENT(IN)  :: PL_STATE
        CLASS(PL_DS_TYPE),    INTENT(IN)  :: PL_DS

        INTEGER, INTENT(IN) :: TIME_IDX
        INTEGER :: STATUS, PL_I, OUT_I, I, NDIMS, DLEN, SPECIES_ID, SLICE_ID, CORNER_ID, COORD_ID, PT_ID, SEG_ID
        INTEGER :: DIMID_TIME, DIMID_SPECIES_PL, DIMID_SEG_ID
        REAL(DP), ALLOCATABLE :: Y_HALF_TMP(:,:,:), Z_HALF_TMP(:,:,:)
        REAL(DP), ALLOCATABLE :: SLICE_POLYS_M_TMP(:,:,:,:)
        CHARACTER(LEN=NF90_MAX_NAME) :: DNAME

        IF (.NOT. PL_OUT%IS_OPEN) STOP "PL_OUT_WRITE: FILE NOT OPEN (CALL INIT FIRST)"
        IF (PL_OUT%NSEG <= 0) STOP "PL_OUT_WRITE: NSEG NOT SET"
        IF (PL_OUT%NSPL <= 0) STOP "PL_OUT_WRITE: NSPL NOT SET"
        IF (PL_OUT%NTOUT <= 0) STOP "PL_OUT_WRITE: NTOUT NOT SET"
        IF (PL_STATE%NSEG /= PL_OUT%NSEG) STOP "PL_OUT_WRITE: PL_STATE NSEG MISMATCH"
        IF (.NOT. ALLOCATED(PL_OUT%TIME_IDX)) STOP "PL_OUT_WRITE: TIME_IDX NOT ALLOCATED (CALL READ_STATIC)"
        ! IF (TIME_IDX < PL_OUT%TIME_IDX(1) .OR. TIME_IDX > PL_OUT%TIME_IDX(PL_OUT%NTOUT)) &
        ! STOP "PL_OUT_WRITE: TIME_IDX OUT OF RANGE"

        ! IF TIME_IDX(I) aligns with PL_OUT%TIME_IDX write PL OUT VARS
        IF (.NOT. ALLOCATED(PL_STATE%PL_MASS)) STOP "PL_OUT_WRITE: PL_STATE%PL_MASS NOT ALLOCATED"
        IF (.NOT. ALLOCATED(PL_STATE%M_FRAC)) STOP "PL_OUT_WRITE: PL_STATE%M_FRAC NOT ALLOCATED"
        IF (.NOT. ALLOCATED(PL_STATE%W_SLICE)) STOP "PL_OUT_WRITE: PL_STATE%W_SLICE NOT ALLOCATED"
        IF (.NOT. ALLOCATED(PL_STATE%Y_HALF)) STOP "PL_OUT_WRITE: PL_STATE%Y_HALF NOT ALLOCATED"
        IF (.NOT. ALLOCATED(PL_STATE%Z_HALF)) STOP "PL_OUT_WRITE: PL_STATE%Z_HALF NOT ALLOCATED"

        IF (.NOT. ALLOCATED(PL_OUT%PL_MASS)) STOP "PL_OUT_WRITE: PL_OUT%PL_MASS NOT ALLOCATED"
        
        IF (PL_DS%OUTPUT_PL_SLICES == 1) THEN
            IF (.NOT. ALLOCATED(PL_OUT%Y_HALF)) STOP "PL_OUT_WRITE: PL_OUT%Y_HALF NOT ALLOCATED"
            IF (.NOT. ALLOCATED(PL_OUT%Z_HALF)) STOP "PL_OUT_WRITE: PL_OUT%Z_HALF NOT ALLOCATED"
            IF (.NOT. ALLOCATED(PL_OUT%M_FRAC)) STOP "PL_OUT_WRITE: PL_OUT%M_FRAC NOT ALLOCATED"
            IF (.NOT. ALLOCATED(PL_OUT%W_SLICE)) STOP "PL_OUT_WRITE: PL_OUT%W_SLICE NOT ALLOCATED"
            IF (.NOT. ALLOCATED(PL_OUT%ELLIPSES_M)) STOP "PL_OUT_WRITE: PL_OUT%ELLIPSES_M NOT ALLOCATED"
            IF (.NOT. ALLOCATED(PL_OUT%SLICE_POLYS_M)) STOP "PL_OUT_WRITE: PL_OUT%SLICE_POLYS_M NOT ALLOCATED"
        END IF

        IF (TIME_IDX < PL_DS%TIME_IDX(1) .OR. TIME_IDX > PL_DS%TIME_IDX(PL_DS%NTPL)) THEN
            RETURN
        END IF

        PL_I = 0
        OUT_I = 0
        DO I = 1, PL_DS%NTPL
            IF (PL_DS%TIME_IDX(I) == TIME_IDX) THEN
                PL_I = I
                EXIT
            END IF
        END DO

        DO I = 1, PL_OUT%NTOUT
            IF (PL_OUT%TIME_IDX(I) == TIME_IDX) THEN
                OUT_I = I
                EXIT
            END IF
        END DO 

        IF (OUT_I == 0) THEN
            RETURN
        END IF

        IF (OUT_I > 0) THEN
            ! WRITE SLICE GEOMETRY IF APPLICABLE
            IF (PL_DS%OUTPUT_PL_SLICES == 1) THEN

                PL_OUT%M_FRAC(:) = PL_STATE%M_FRAC(:)
                PL_OUT%W_SLICE(:) = PL_STATE%W_SLICE(:)

                DO PT_ID = 1, PL_DS%NPOINTS
                    DO COORD_ID = 1, 3
                        PL_OUT%ELLIPSES_M(:,PT_ID,COORD_ID,OUT_I) = PL_STATE%ELLIPSES_M(:,PT_ID,COORD_ID)
                    END DO
                END DO

                DO SLICE_ID = 1, PL_OUT%NSLICES
                    PL_OUT%Y_HALF(:,SLICE_ID,OUT_I) = 0.0_DP
                    PL_OUT%Z_HALF(:,SLICE_ID,OUT_I) = 0.0_DP
                    PL_OUT%SLICE_POLYS_M(:,SLICE_ID,:,:,OUT_I) = 0.0_DP

                    PL_OUT%Y_HALF(:,SLICE_ID,OUT_I) = PL_STATE%Y_HALF(:,SLICE_ID)
                    PL_OUT%Z_HALF(:,SLICE_ID,OUT_I) = PL_STATE%Z_HALF(:,SLICE_ID)
                    PL_OUT%SLICE_POLYS_M(:,SLICE_ID,:,:,OUT_I) = PL_STATE%SLICE_POLYS_M(:,SLICE_ID,:,:)
                END DO

            END IF

            PL_OUT%PL_MASS(:,:,OUT_I) = 0.0_DP
            DO SPECIES_ID = 1, PL_STATE%NSPL
                IF (ANY(PL_DS%SPECIES_EMI_NUM(:) == PL_STATE%SPECIES_PL_NUM(SPECIES_ID))) THEN
                    PL_OUT%PL_MASS(:,SPECIES_ID,OUT_I) = PL_STATE%PL_MASS(:,SPECIES_ID)
                    ! DO SEG_ID = 1, PL_OUT%NSEG
                    !     PRINT *, "PL_OUT PL_MASS=", PL_OUT%PL_MASS(SEG_ID,SPECIES_ID,OUT_I), " FOR SPECIES ", &
                    !             PL_STATE%SPECIES_PL_NUM(SPECIES_ID), "SEG ", SEG_ID, " AT TIME_IDX ", TIME_IDX
                    ! END DO
                END IF
            END DO
        
            DO SPECIES_ID = 1, PL_OUT%NSPL
                STATUS = NF90_PUT_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_PL_MASS, PL_OUT%PL_MASS(:, SPECIES_ID, OUT_I), &
                                START=[OUT_I, SPECIES_ID, 1], COUNT=[1, 1, PL_OUT%NSEG])
                CALL NC_CHECK(STATUS, "NF90_PUT_VAR(pl_mass)")

            END DO

            ! WRITE SLICE GEOMETRY IF APPLICABLE
            IF (PL_DS%OUTPUT_PL_SLICES == 1) THEN
                DO SLICE_ID = 1, PL_OUT%NSLICES
                    
                    STATUS = NF90_PUT_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_M_FRAC, PL_OUT%M_FRAC(:), START=[SLICE_ID], COUNT=[1])
                    CALL NC_CHECK(STATUS, "NF90_PUT_VAR(m_frac)")

                    STATUS = NF90_PUT_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_W_SLICE, PL_OUT%W_SLICE(:), START=[SLICE_ID], COUNT=[1])
                    CALL NC_CHECK(STATUS, "NF90_PUT_VAR(w_slice)")

                    STATUS = NF90_PUT_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_Y_HALF, PL_OUT%Y_HALF(:,SLICE_ID,OUT_I), &
                                    START=[OUT_I, SLICE_ID, 1], COUNT=[1, 1, PL_OUT%NSEG])
                    CALL NC_CHECK(STATUS, "NF90_PUT_VAR(y_half)")

                    STATUS = NF90_PUT_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_Z_HALF, PL_OUT%Z_HALF(:,SLICE_ID,OUT_I), &
                                    START=[OUT_I, SLICE_ID, 1], COUNT=[1, 1, PL_OUT%NSEG])
                    CALL NC_CHECK(STATUS, "NF90_PUT_VAR(z_half)")

                    ! WRITE SLICE POLYGONS
                    DO CORNER_ID = 1, 4
                        DO COORD_ID = 1, 3
                            STATUS = NF90_PUT_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_SLICE_POLYS_M, &
                                    PL_OUT%SLICE_POLYS_M(:,SLICE_ID,CORNER_ID,COORD_ID,OUT_I), & 
                                    START=[OUT_I, COORD_ID, CORNER_ID, SLICE_ID, 1], COUNT=[1, 1, 1, 1, PL_OUT%NSEG])
                            CALL NC_CHECK(STATUS, "NF90_PUT_VAR(slice_polys_m)")
                        END DO
                    END DO
                END DO
                
                ! WRITE ELLIPSE POINTS
                DO PT_ID = 1, PL_DS%NPOINTS
                    DO COORD_ID = 1, 3
                        STATUS = NF90_PUT_VAR(PL_OUT%PL_OUT_NCID, PL_OUT%VARID_ELLIPSES_M, &
                                PL_OUT%ELLIPSES_M(:,PT_ID,COORD_ID,OUT_I), & 
                                START=[OUT_I, COORD_ID, PT_ID, 1], COUNT=[1, 1, 1, PL_OUT%NSEG])
                        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(ellipses_m)")
                    END DO
                END DO  
            END IF
        END IF

    END SUBROUTINE PL_OUT_WRITE

    SUBROUTINE PL_OUT_CLOSE(PL_OUT)
        CLASS(PL_OUT_TYPE), INTENT(INOUT) :: PL_OUT
        INTEGER :: STATUS, I

        IF (PL_OUT%IS_OPEN) THEN
            STATUS = NF90_CLOSE(PL_OUT%PL_OUT_NCID)
            CALL NC_CHECK(STATUS, "NF90_CLOSE(PL_OUT) IN PL_OUT_CLOSE")
            PL_OUT%IS_OPEN = .FALSE.
        END IF

    END SUBROUTINE PL_OUT_CLOSE

    ! ... BOXM_OUT METHODS ...
    SUBROUTINE BOXM_OUT_INIT(BOXM_OUT, FILEPATH)
        CLASS(BOXM_OUT_TYPE), INTENT(INOUT) :: BOXM_OUT
        CHARACTER(LEN=*),  INTENT(IN)    :: FILEPATH

        INTEGER  :: STATUS

        IF (BOXM_OUT%IS_OPEN) THEN
            STATUS = NF90_CLOSE(BOXM_OUT%BOXM_OUT_NCID)
            CALL NC_CHECK(STATUS, "NF90_CLOSE(BOXM_OUT) IN BOXM_OUT_INIT")
        END IF

        STATUS = NF90_OPEN(TRIM(FILEPATH), NF90_WRITE, BOXM_OUT%BOXM_OUT_NCID)
        CALL NC_CHECK(STATUS, "NF90_OPEN(BOXM_OUT)")

        BOXM_OUT%IS_OPEN = .TRUE.

        ! DIM IDS AND LENGTHS
        STATUS = NF90_INQ_DIMID(BOXM_OUT%BOXM_OUT_NCID, "cell", BOXM_OUT%DIMID_CELL)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(cell)")

        STATUS = NF90_INQUIRE_DIMENSION(BOXM_OUT%BOXM_OUT_NCID, BOXM_OUT%DIMID_CELL, LEN=BOXM_OUT%NCELL)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(cell)")

        STATUS = NF90_INQ_DIMID(BOXM_OUT%BOXM_OUT_NCID, "species_out", BOXM_OUT%DIMID_SPECIES_OUT)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(species_out)")

        STATUS = NF90_INQUIRE_DIMENSION(BOXM_OUT%BOXM_OUT_NCID, BOXM_OUT%DIMID_SPECIES_OUT, LEN=BOXM_OUT%NSOUT)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(species_out)")

        STATUS = NF90_INQ_DIMID(BOXM_OUT%BOXM_OUT_NCID, "time", BOXM_OUT%DIMID_TIME)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(time)")

        STATUS = NF90_INQUIRE_DIMENSION(BOXM_OUT%BOXM_OUT_NCID, BOXM_OUT%DIMID_TIME, LEN=BOXM_OUT%NTOUT)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(time)")

        ! VAR IDS
        STATUS = NF90_INQ_VARID(BOXM_OUT%BOXM_OUT_NCID, "time_rel_s", BOXM_OUT%VARID_TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_rel_s)")

        STATUS = NF90_INQ_VARID(BOXM_OUT%BOXM_OUT_NCID, "time_idx", BOXM_OUT%VARID_TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_idx)")

        STATUS = NF90_INQ_VARID(BOXM_OUT%BOXM_OUT_NCID, "species_out_num", BOXM_OUT%VARID_SPECIES_OUT_NUM)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(species_out_num)")

        STATUS = NF90_INQ_VARID(BOXM_OUT%BOXM_OUT_NCID, "Y_bg_c", BOXM_OUT%VARID_Y_BG_C)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(Y_bg_c)")

        STATUS = NF90_INQ_VARID(BOXM_OUT%BOXM_OUT_NCID, "Y_del_c", BOXM_OUT%VARID_Y_DEL_C)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(Y_del_c)")

        STATUS = NF90_INQ_VARID(BOXM_OUT%BOXM_OUT_NCID, "active_flag", BOXM_OUT%VARID_ACTIVE_FLAG)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(active_flag)")

    END SUBROUTINE BOXM_OUT_INIT

    SUBROUTINE BOXM_OUT_READ_STATIC(BOXM_OUT)
        CLASS(BOXM_OUT_TYPE), INTENT(INOUT) :: BOXM_OUT
        INTEGER :: STATUS

        IF (.NOT. BOXM_OUT%IS_OPEN) STOP "BOXM_OUT_READ_STATIC: FILE NOT OPEN (CALL INIT FIRST)"
        IF (BOXM_OUT%NCELL <= 0) STOP "BOXM_OUT_READ_STATIC: NCELL NOT SET"
        IF (BOXM_OUT%NSOUT <= 0) STOP "BOXM_OUT_READ_STATIC: NSOUT NOT SET"
        IF (BOXM_OUT%NTOUT <= 0) STOP "BOXM_OUT_READ_STATIC: NTOUT NOT SET"

        IF (.NOT. ALLOCATED(BOXM_OUT%TIME_REL_S)) ALLOCATE(BOXM_OUT%TIME_REL_S(BOXM_OUT%NTOUT))
        IF (.NOT. ALLOCATED(BOXM_OUT%TIME_IDX)) ALLOCATE(BOXM_OUT%TIME_IDX(BOXM_OUT%NTOUT))
        IF (.NOT. ALLOCATED(BOXM_OUT%SPECIES_OUT_NUM)) ALLOCATE(BOXM_OUT%SPECIES_OUT_NUM(BOXM_OUT%NSOUT))
        IF (.NOT. ALLOCATED(BOXM_OUT%Y_BG_C)) ALLOCATE(BOXM_OUT%Y_BG_C(BOXM_OUT%NCELL, BOXM_OUT%NSOUT, BOXM_OUT%NTOUT))
        IF (.NOT. ALLOCATED(BOXM_OUT%Y_DEL_C)) ALLOCATE(BOXM_OUT%Y_DEL_C(BOXM_OUT%NCELL, BOXM_OUT%NSOUT, BOXM_OUT%NTOUT))
        IF (.NOT. ALLOCATED(BOXM_OUT%ACTIVE_FLAG)) ALLOCATE(BOXM_OUT%ACTIVE_FLAG(BOXM_OUT%NCELL, BOXM_OUT%NTOUT))

        STATUS = NF90_GET_VAR(BOXM_OUT%BOXM_OUT_NCID, BOXM_OUT%VARID_TIME_REL_S, BOXM_OUT%TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_rel_s)")

        STATUS = NF90_GET_VAR(BOXM_OUT%BOXM_OUT_NCID, BOXM_OUT%VARID_TIME_IDX, BOXM_OUT%TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_idx)")

        STATUS = NF90_GET_VAR(BOXM_OUT%BOXM_OUT_NCID, BOXM_OUT%VARID_SPECIES_OUT_NUM, BOXM_OUT%SPECIES_OUT_NUM)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(species_out_num)")

    END SUBROUTINE BOXM_OUT_READ_STATIC

    SUBROUTINE BOXM_OUT_WRITE(BOXM_OUT, BOXM_STATE, TIME_IDX)
        CLASS(BOXM_OUT_TYPE), INTENT(INOUT) :: BOXM_OUT
        CLASS(BOXM_STATE_TYPE), INTENT(IN)  :: BOXM_STATE

        INTEGER, INTENT(IN) :: TIME_IDX
        INTEGER :: STATUS, OUT_I, I, SPECIES_ID, BOXM_ID_S
        INTEGER, ALLOCATABLE :: ACTIVE_FLAG_TMP(:,:)

        ! Allocate temporary arrays for output
        REAL(DP), ALLOCATABLE :: Y_BG_C_OUT(:), Y_DEL_C_OUT(:)
        
        IF (.NOT. BOXM_OUT%IS_OPEN) STOP "BOXM_OUT_WRITE: FILE NOT OPEN (CALL INIT FIRST)"
        IF (BOXM_OUT%NCELL <= 0) STOP "BOXM_OUT_WRITE: NCELL NOT SET"
        IF (BOXM_OUT%NSOUT <= 0) STOP "BOXM_OUT_WRITE: NSOUT NOT SET"
        IF (BOXM_OUT%NTOUT <= 0) STOP "BOXM_OUT_WRITE: NTOUT NOT SET"
        IF (.NOT. ALLOCATED(BOXM_OUT%TIME_IDX)) STOP "BOXM_OUT_WRITE: TIME_IDX NOT ALLOCATED (CALL READ_STATIC)"

        ! WRITE BOXM OUT VARS
        IF (.NOT. ALLOCATED(BOXM_STATE%Y_BG_C)) STOP "BOXM_OUT_WRITE: BOXM_STATE%Y_BG_C NOT ALLOCATED"
        IF (.NOT. ALLOCATED(BOXM_STATE%Y_DEL_C)) STOP "BOXM_OUT_WRITE: BOXM_STATE%Y_DEL_C NOT ALLOCATED"
        IF (.NOT. ALLOCATED(BOXM_STATE%ACTIVE_FLAG)) STOP "BOXM_OUT_WRITE: BOXM_STATE%ACTIVE_FLAG NOT ALLOCATED"


        ! Find output index that matches TIME_IDX exactly (like PL_OUT_WRITE)
        OUT_I = 0
        DO I = 1, BOXM_OUT%NTOUT
            IF (BOXM_OUT%TIME_IDX(I) == TIME_IDX) THEN
                OUT_I = I
                EXIT
            END IF
        END DO

        IF (OUT_I == 0) THEN
            RETURN
        END IF

        ! Print summary for all species
        DO SPECIES_ID = 1, BOXM_OUT%NSOUT
            IF (ALLOCATED(BOXM_OUT%SPECIES_OUT_NUM)) THEN
                BOXM_ID_S = BOXM_OUT%SPECIES_OUT_NUM(SPECIES_ID)
            ELSE
                BOXM_ID_S = SPECIES_ID
            END IF
        
        END DO

        BOXM_OUT%ACTIVE_FLAG(:, OUT_I) = MERGE(1, 0, BOXM_STATE%ACTIVE_FLAG)

        ALLOCATE(Y_BG_C_OUT(BOXM_OUT%NCELL))
        ALLOCATE(Y_DEL_C_OUT(BOXM_OUT%NCELL))
        DO SPECIES_ID = 1, BOXM_OUT%NSOUT
            IF (ALLOCATED(BOXM_OUT%SPECIES_OUT_NUM)) THEN
                BOXM_ID_S = BOXM_OUT%SPECIES_OUT_NUM(SPECIES_ID)
            ELSE
                BOXM_ID_S = SPECIES_ID
            END IF
            IF (BOXM_ID_S >= 1 .AND. BOXM_ID_S <= BOXM_STATE%NSBOXM) THEN
                Y_BG_C_OUT(:) = BOXM_STATE%Y_BG_C(:, BOXM_ID_S) * 1.0E+09 / BOXM_STATE%M(:) ! Convert from MOL/CM3 to ppb
                Y_DEL_C_OUT(:) = BOXM_STATE%Y_DEL_C(:, BOXM_ID_S) * 1.0E+09 / BOXM_STATE%M(:) ! Convert from MOL/CM3 to ppb
            ELSE
                Y_BG_C_OUT(:) = 0.0_DP
                Y_DEL_C_OUT(:) = 0.0_DP
            END IF
            STATUS = NF90_PUT_VAR(BOXM_OUT%BOXM_OUT_NCID, BOXM_OUT%VARID_Y_BG_C, Y_BG_C_OUT, &
                START=[OUT_I, SPECIES_ID, 1], COUNT=[1, 1, BOXM_OUT%NCELL])
            CALL NC_CHECK(STATUS, "NF90_PUT_VAR(Y_bg_c)")
            STATUS = NF90_PUT_VAR(BOXM_OUT%BOXM_OUT_NCID, BOXM_OUT%VARID_Y_DEL_C, Y_DEL_C_OUT, &
                START=[OUT_I, SPECIES_ID, 1], COUNT=[1, 1, BOXM_OUT%NCELL])
            CALL NC_CHECK(STATUS, "NF90_PUT_VAR(Y_del_c)")
        END DO
        DEALLOCATE(Y_BG_C_OUT)
        DEALLOCATE(Y_DEL_C_OUT)
        
        ALLOCATE(ACTIVE_FLAG_TMP(BOXM_OUT%NCELL, 1))
        ACTIVE_FLAG_TMP(:, 1) = MERGE(1, 0, BOXM_STATE%ACTIVE_FLAG)

        STATUS = NF90_PUT_VAR(BOXM_OUT%BOXM_OUT_NCID, BOXM_OUT%VARID_ACTIVE_FLAG, ACTIVE_FLAG_TMP, &
                            START=[OUT_I, 1], COUNT=[1, BOXM_OUT%NCELL])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(active_flag)")
        DEALLOCATE(ACTIVE_FLAG_TMP)

    END SUBROUTINE BOXM_OUT_WRITE

    SUBROUTINE BOXM_OUT_CLOSE(BOXM_OUT)
        CLASS(BOXM_OUT_TYPE), INTENT(INOUT) :: BOXM_OUT
        INTEGER :: STATUS

        IF (BOXM_OUT%IS_OPEN) THEN
            STATUS = NF90_CLOSE(BOXM_OUT%BOXM_OUT_NCID)
            CALL NC_CHECK(STATUS, "NF90_CLOSE(BOXM_OUT) IN BOXM_OUT_CLOSE")
            BOXM_OUT%IS_OPEN = .FALSE.
        END IF

    END SUBROUTINE BOXM_OUT_CLOSE

    ! ... PATCH_TABLE METHODS ...
    SUBROUTINE PATCH_TABLE_INIT(PATCH_TABLE, FILEPATH)
        CLASS(PATCH_TABLE_TYPE), INTENT(INOUT) :: PATCH_TABLE
        CHARACTER(LEN=*),  INTENT(IN)    :: FILEPATH
        
        INTEGER  :: STATUS

        IF (PATCH_TABLE%IS_OPEN) THEN
            STATUS = NF90_CLOSE(PATCH_TABLE%PATCH_TABLE_NCID)
            CALL NC_CHECK(STATUS, "NF90_CLOSE(PATCH_TABLE) IN PATCH_TABLE_INIT")
        END IF

        STATUS = NF90_OPEN(TRIM(FILEPATH), NF90_WRITE, PATCH_TABLE%PATCH_TABLE_NCID)
        CALL NC_CHECK(STATUS, "NF90_OPEN(PATCH_TABLE)")

        PATCH_TABLE%IS_OPEN = .TRUE.

        ! DIM IDS AND LENGTHS
        STATUS = NF90_INQ_DIMID(PATCH_TABLE%PATCH_TABLE_NCID, "row", PATCH_TABLE%DIMID_ROW)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(row)")

        STATUS = NF90_INQUIRE_DIMENSION(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%DIMID_ROW, LEN=PATCH_TABLE%NROWS)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(row)")

        STATUS = NF90_INQ_DIMID(PATCH_TABLE%PATCH_TABLE_NCID, "species_out", PATCH_TABLE%DIMID_SPECIES_OUT)
        CALL NC_CHECK(STATUS, "NF90_INQ_DIMID(species_out)")

        STATUS = NF90_INQUIRE_DIMENSION(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%DIMID_SPECIES_OUT, LEN=PATCH_TABLE%NSOUT)
        CALL NC_CHECK(STATUS, "NF90_INQUIRE_DIMENSION(species_out)")

        PATCH_TABLE%NEXT_ROW = PATCH_TABLE%NROWS + 1

        ! VAR IDS
        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "row", PATCH_TABLE%VARID_ROW)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(row)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "species_out_num", PATCH_TABLE%VARID_SPECIES_OUT_NUM)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(species_out_num)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "row_cell_c", PATCH_TABLE%VARID_ROW_CELL_C)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(row_cell_c)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "row_cell_f", PATCH_TABLE%VARID_ROW_CELL_F)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(row_cell_f)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "time_rel_s", PATCH_TABLE%VARID_TIME_REL_S)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_rel_s)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "time_idx", PATCH_TABLE%VARID_TIME_IDX)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(time_idx)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "latitude_f", PATCH_TABLE%VARID_LATITUDE_F)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(latitude_f)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "longitude_f", PATCH_TABLE%VARID_LONGITUDE_F)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(longitude_f)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "altitude_f", PATCH_TABLE%VARID_ALTITUDE_F)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(altitude_f)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "level_f", PATCH_TABLE%VARID_LEVEL_F)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(level_f)")

        STATUS = NF90_INQ_VARID(PATCH_TABLE%PATCH_TABLE_NCID, "Y_del_f", PATCH_TABLE%VARID_Y_DEL_F)
        CALL NC_CHECK(STATUS, "NF90_INQ_VARID(Y_del_f)")

    END SUBROUTINE PATCH_TABLE_INIT

    SUBROUTINE PATCH_TABLE_READ_STATIC(PATCH_TABLE)
        CLASS(PATCH_TABLE_TYPE), INTENT(INOUT) :: PATCH_TABLE
        INTEGER :: STATUS

        IF (.NOT. PATCH_TABLE%IS_OPEN) STOP "PATCH_TABLE_READ_STATIC: FILE NOT OPEN (CALL INIT FIRST)"
        IF (PATCH_TABLE%NSOUT <= 0) STOP "PATCH_TABLE_READ_STATIC: NSOUT NOT SET"

        IF (.NOT. ALLOCATED(PATCH_TABLE%SPECIES_OUT_NUM)) ALLOCATE(PATCH_TABLE%SPECIES_OUT_NUM(PATCH_TABLE%NSOUT))
        IF (.NOT. ALLOCATED(PATCH_TABLE%TIME_REL_S)) ALLOCATE(PATCH_TABLE%TIME_REL_S(PATCH_TABLE%NROWS))
        IF (.NOT. ALLOCATED(PATCH_TABLE%TIME_IDX)) ALLOCATE(PATCH_TABLE%TIME_IDX(PATCH_TABLE%NROWS))
        IF (.NOT. ALLOCATED(PATCH_TABLE%ROW_CELL_C)) ALLOCATE(PATCH_TABLE%ROW_CELL_C(PATCH_TABLE%NROWS))
        IF (.NOT. ALLOCATED(PATCH_TABLE%ROW_CELL_F)) ALLOCATE(PATCH_TABLE%ROW_CELL_F(PATCH_TABLE%NROWS))
        
        STATUS = NF90_GET_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_SPECIES_OUT_NUM, PATCH_TABLE%SPECIES_OUT_NUM)
        CALL NC_CHECK(STATUS, "NF90_GET_VAR(species_out_num)")

        IF (PATCH_TABLE%NROWS > 0) THEN
            STATUS = NF90_GET_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_TIME_REL_S, PATCH_TABLE%TIME_REL_S)
            CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_rel_s)")

            STATUS = NF90_GET_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_TIME_IDX, PATCH_TABLE%TIME_IDX)
            CALL NC_CHECK(STATUS, "NF90_GET_VAR(time_idx)")

            STATUS = NF90_GET_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_ROW_CELL_C, PATCH_TABLE%ROW_CELL_C)
            CALL NC_CHECK(STATUS, "NF90_GET_VAR(row_cell_c)")

            STATUS = NF90_GET_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_ROW_CELL_F, PATCH_TABLE%ROW_CELL_F)
            CALL NC_CHECK(STATUS, "NF90_GET_VAR(row_cell_f)")
        END IF

    END SUBROUTINE PATCH_TABLE_READ_STATIC

    SUBROUTINE PATCH_TABLE_WRITE(PATCH_TABLE, PATCH_STATE, BOXM_DS, BOXM_STATE, TIME_IDX)
        CLASS(PATCH_TABLE_TYPE), INTENT(INOUT) :: PATCH_TABLE
        CLASS(PATCH_STATE_TYPE), INTENT(IN) :: PATCH_STATE
        CLASS(BOXM_DS_TYPE), INTENT(IN) :: BOXM_DS
        CLASS(BOXM_STATE_TYPE), INTENT(IN) :: BOXM_STATE
        
        INTEGER, INTENT(IN) :: TIME_IDX
        INTEGER :: STATUS, OUT_I, I, ROW_IDX, CELL_C, CELL_F
        INTEGER :: NX_F, NY_F, NZ_F, IX_F, IY_F, IZ_F, REM_F
        INTEGER :: SPECIES_ID, BOXM_ID_S
        INTEGER, ALLOCATABLE :: ROW_IDS(:)
        INTEGER, ALLOCATABLE :: TIME_IDX_ROWS(:)
        REAL(DP), ALLOCATABLE :: LAT_ROWS(:), LON_ROWS(:), ALT_ROWS(:)
        REAL(DP), ALLOCATABLE :: Y_DEL_F_OUT(:,:)
        REAL(DP) :: HRES_F_DEG, DZ_F
        REAL(DP) :: LAT_C, LON_C, ALT_C

        IF (.NOT. PATCH_TABLE%IS_OPEN) STOP "PATCH_TABLE_WRITE: FILE NOT OPEN (CALL INIT FIRST)"
        IF (PATCH_TABLE%NSOUT <= 0) STOP "PATCH_TABLE_WRITE: NSOUT NOT SET"

        ! WRITE PATCH TABLE VARS
        IF (.NOT. ALLOCATED(PATCH_STATE%TIME_REL_S)) RETURN
        IF (.NOT. ALLOCATED(PATCH_STATE%Y_DEL_F)) RETURN
        IF (PATCH_STATE%NROWS <= 0) RETURN
        IF (PATCH_STATE%ROW_IDX <= 0) RETURN

        OUT_I = PATCH_TABLE%NEXT_ROW
        NX_F = MAX(1, NINT(BOXM_DS%HRES_SIM_C / BOXM_DS%HRES_SIM_F))
        NY_F = NX_F
        NZ_F = MAX(1, NINT(BOXM_DS%VRES_SIM_C / BOXM_DS%VRES_SIM_F))
        HRES_F_DEG = BOXM_DS%HRES_SIM_C / REAL(NX_F, DP)
        DZ_F = BOXM_DS%VRES_SIM_C / REAL(NZ_F, DP)

        ALLOCATE(ROW_IDS(PATCH_STATE%NROWS))
        ALLOCATE(TIME_IDX_ROWS(PATCH_STATE%NROWS))
        ALLOCATE(LAT_ROWS(PATCH_STATE%NROWS))
        ALLOCATE(LON_ROWS(PATCH_STATE%NROWS))
        ALLOCATE(ALT_ROWS(PATCH_STATE%NROWS))
        ROW_IDS(:) = [(OUT_I + I - 1, I=1, PATCH_STATE%NROWS)]
        TIME_IDX_ROWS(:) = TIME_IDX

        DO ROW_IDX = 1, PATCH_STATE%NROWS
            CELL_C = PATCH_STATE%ROW_CELL_C(ROW_IDX)
            CELL_F = PATCH_STATE%ROW_CELL_F(ROW_IDX)

            LAT_C = BOXM_STATE%LATITUDE_C(CELL_C)
            LON_C = BOXM_STATE%LONGITUDE_C(CELL_C)
            ALT_C = BOXM_STATE%ALTITUDE_C(CELL_C)

            IZ_F = (CELL_F - 1) / (NX_F * NY_F) + 1
            REM_F = MOD(CELL_F - 1, NX_F * NY_F)
            IY_F = REM_F / NX_F + 1
            IX_F = MOD(REM_F, NX_F) + 1

            LAT_ROWS(ROW_IDX) = LAT_C - 0.5_DP * BOXM_DS%HRES_SIM_C + (REAL(IY_F, DP) - 0.5_DP) * HRES_F_DEG
            LON_ROWS(ROW_IDX) = LON_C - 0.5_DP * BOXM_DS%HRES_SIM_C + (REAL(IX_F, DP) - 0.5_DP) * HRES_F_DEG
            ALT_ROWS(ROW_IDX) = ALT_C - 0.5_DP * BOXM_DS%VRES_SIM_C + (REAL(IZ_F, DP) - 0.5_DP) * DZ_F
        END DO

        STATUS = NF90_PUT_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_ROW, ROW_IDS, &
                START=[OUT_I], COUNT=[PATCH_STATE%NROWS])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(row)")

        STATUS = NF90_PUT_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_TIME_REL_S, PATCH_STATE%TIME_REL_S, &
                        START=[OUT_I], COUNT=[PATCH_STATE%NROWS])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(time_rel_s)")

        STATUS = NF90_PUT_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_TIME_IDX, TIME_IDX_ROWS, &
                        START=[OUT_I], COUNT=[PATCH_STATE%NROWS])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(time_idx)")

        STATUS = NF90_PUT_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_ROW_CELL_C, PATCH_STATE%ROW_CELL_C, &
                START=[OUT_I], COUNT=[PATCH_STATE%NROWS])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(row_cell_c)")

        STATUS = NF90_PUT_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_ROW_CELL_F, PATCH_STATE%ROW_CELL_F, &
                START=[OUT_I], COUNT=[PATCH_STATE%NROWS])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(row_cell_f)")

        STATUS = NF90_PUT_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_LATITUDE_F, LAT_ROWS, &
                START=[OUT_I], COUNT=[PATCH_STATE%NROWS])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(latitude_f)")

        STATUS = NF90_PUT_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_LONGITUDE_F, LON_ROWS, &
                START=[OUT_I], COUNT=[PATCH_STATE%NROWS])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(longitude_f)")

        STATUS = NF90_PUT_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_ALTITUDE_F, ALT_ROWS, &
                START=[OUT_I], COUNT=[PATCH_STATE%NROWS])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(altitude_f)")

        ! Build correctly transposed output array: (NSOUT, NROWS) in Fortran col-major
        ! so that species varies fastest, matching the netcdf variable layout (species_out, row).
        ! PATCH_STATE%Y_DEL_F is (NROWS, NSBOXM); we must select the right BOXM column per
        ! output species using PATCH_TABLE%SPECIES_OUT_NUM.
        ALLOCATE(Y_DEL_F_OUT(PATCH_TABLE%NSOUT, PATCH_STATE%NROWS))
        Y_DEL_F_OUT(:,:) = 0.0_DP
        IF (ALLOCATED(PATCH_TABLE%SPECIES_OUT_NUM)) THEN
            DO SPECIES_ID = 1, PATCH_TABLE%NSOUT
                BOXM_ID_S = PATCH_TABLE%SPECIES_OUT_NUM(SPECIES_ID)
                IF (BOXM_ID_S >= 1 .AND. BOXM_ID_S <= PATCH_STATE%NSBOXM) THEN
                    Y_DEL_F_OUT(SPECIES_ID, :) = PATCH_STATE%Y_DEL_F(:, BOXM_ID_S) * &
                    1.0E+09 / BOXM_STATE%M(PATCH_STATE%ROW_CELL_C(:))   ! Convert from MOL/CM3 to ppb
                END IF
            END DO
        END IF

        STATUS = NF90_PUT_VAR(PATCH_TABLE%PATCH_TABLE_NCID, PATCH_TABLE%VARID_Y_DEL_F, Y_DEL_F_OUT, &
                        START=[1, OUT_I], COUNT=[PATCH_TABLE%NSOUT, PATCH_STATE%NROWS])
        CALL NC_CHECK(STATUS, "NF90_PUT_VAR(Y_del_f)")
        DEALLOCATE(Y_DEL_F_OUT)

        PATCH_TABLE%NEXT_ROW = PATCH_TABLE%NEXT_ROW + PATCH_STATE%NROWS
        PATCH_TABLE%NROWS = PATCH_TABLE%NEXT_ROW - 1

        DEALLOCATE(ROW_IDS)
        DEALLOCATE(TIME_IDX_ROWS)
        DEALLOCATE(LAT_ROWS)
        DEALLOCATE(LON_ROWS)
        DEALLOCATE(ALT_ROWS)

    END SUBROUTINE PATCH_TABLE_WRITE

    SUBROUTINE PATCH_TABLE_CLOSE(PATCH_TABLE)
        CLASS(PATCH_TABLE_TYPE), INTENT(INOUT) :: PATCH_TABLE
        INTEGER :: STATUS

        IF (PATCH_TABLE%IS_OPEN) THEN
            STATUS = NF90_CLOSE(PATCH_TABLE%PATCH_TABLE_NCID)
            CALL NC_CHECK(STATUS, "NF90_CLOSE(PATCH_TABLE) IN PATCH_TABLE_CLOSE")
            PATCH_TABLE%IS_OPEN = .FALSE.
        END IF

    END SUBROUTINE PATCH_TABLE_CLOSE

END MODULE DEFINE_OUTPUT_TYPES

MODULE BOXM_RUN_UTILS
    USE RUN_CHEM_UTILS
    IMPLICIT NONE
    
    PRIVATE
    PUBLIC :: BOXM_RUN_INIT, ADVANCE_MET, PROJECT_PLUMES_TO_GRID, RUN_CHEM, BACKPROJECT_GRID_TO_PLUMES
    PUBLIC :: WRITE_OUTPUTS, CLOSE_DATASETS, RESET_STATES
    
CONTAINS
    SUBROUTINE BOXM_RUN_INIT(FL_DS, PL_DS, BOXM_DS, PL_STATE, BOXM_STATE, PATCH_STATE, &
                            PL_OUT, BOXM_OUT, PATCH_TABLE, JOB_ID, DATA_PATH)
        USE HELPERS
        USE RUN_CHEM_UTILS
        USE DEFINE_INPUT_TYPES
        USE DEFINE_STATE_TYPES
        USE DEFINE_OUTPUT_TYPES
        IMPLICIT NONE

        CLASS(FL_DS_TYPE),       INTENT(INOUT) :: FL_DS
        CLASS(PL_DS_TYPE),       INTENT(INOUT) :: PL_DS
        CLASS(BOXM_DS_TYPE),     INTENT(INOUT) :: BOXM_DS

        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(BOXM_STATE_TYPE),  INTENT(INOUT) :: BOXM_STATE
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE

        CLASS(PL_OUT_TYPE),      INTENT(INOUT) :: PL_OUT
        CLASS(BOXM_OUT_TYPE),    INTENT(INOUT) :: BOXM_OUT
        CLASS(PATCH_TABLE_TYPE), INTENT(INOUT) :: PATCH_TABLE

        CHARACTER(LEN=*), INTENT(IN) :: JOB_ID
        CHARACTER(LEN=*), INTENT(IN) :: DATA_PATH
        INTEGER :: SEG_ID

        ! INITIALIZE AND READ INPUT DATASETS
        CALL BOXM_DS%INIT(TRIM(DATA_PATH)//'inputs/'//TRIM(JOB_ID)//'/boxm_ds.nc')
        CALL BOXM_DS%READ_STATIC()
        CALL BOXM_DS%SUMMARY()
        
        IF (BOXM_DS%N_AC > 0) THEN
            CALL FL_DS%INIT(TRIM(DATA_PATH)//'inputs/'//TRIM(JOB_ID)//'/fl_ds.nc')
            CALL FL_DS%READ_STATIC()
            CALL FL_DS%SUMMARY()

            CALL PL_DS%INIT(TRIM(DATA_PATH)//'inputs/'//TRIM(JOB_ID)//'/pl_ds.nc')
            CALL PL_DS%READ_STATIC()
            CALL PL_DS%SUMMARY()
        END IF

        IF (BOXM_DS%N_AC > 0) THEN
            ! STATE TYPE INITIALIZATION FROM INPUT DATASETS
            CALL PL_STATE%INIT_FROM_PL_DS(PL_DS)
            CALL PL_STATE%INIT_FROM_PL_OUT(PL_DS, SEG_ID=1)
            CALL PATCH_STATE%INIT_FROM_BOXM_DS(BOXM_DS)
        END IF
        CALL BOXM_STATE%INIT_FROM_BOXM_DS(BOXM_DS)
        
        IF (BOXM_DS%N_AC > 0) THEN
            ! INITIALISE OUTPUT DATASETS
            CALL PL_OUT%INIT(PL_DS, TRIM(DATA_PATH)//'outputs/'//TRIM(JOB_ID)//'/pl_out.nc')
            CALL PL_OUT%READ_STATIC(PL_DS)

            ! Align plume state species with PL_OUT species list
            PL_STATE%NSPL = PL_OUT%NSPL
            IF (ALLOCATED(PL_STATE%SPECIES_PL_NUM)) DEALLOCATE(PL_STATE%SPECIES_PL_NUM)
            ALLOCATE(PL_STATE%SPECIES_PL_NUM(PL_STATE%NSPL))
            PL_STATE%SPECIES_PL_NUM = PL_OUT%SPECIES_PL_NUM

            IF (ALLOCATED(PL_STATE%PL_MASS)) DEALLOCATE(PL_STATE%PL_MASS)
            ALLOCATE(PL_STATE%PL_MASS(PL_STATE%NSEG, PL_STATE%NSPL))
            PL_STATE%PL_MASS(:,:) = 0.0_DP

            CALL PATCH_TABLE%INIT(TRIM(DATA_PATH)//'outputs/'//TRIM(JOB_ID)//'/patch_table.nc')
            CALL PATCH_TABLE%READ_STATIC()
        END IF

        CALL BOXM_OUT%INIT(TRIM(DATA_PATH)//'outputs/'//TRIM(JOB_ID)//'/boxm_out.nc')
        CALL BOXM_OUT%READ_STATIC()

        IF (BOXM_DS%RUN_CHEM == 1) THEN
            CALL CHEM_ALLOC(BOXM_DS%NCELL, BOXM_DS%NSBOXM, BOXM_DS%NPP, BOXM_DS%NPC, BOXM_DS%NTC, BOXM_DS%NFL)
        END IF

    END SUBROUTINE BOXM_RUN_INIT

    SUBROUTINE PROJECT_PLUMES_TO_GRID(FL_DS, PL_DS, BOXM_DS, PL_STATE, BOXM_STATE, PATCH_STATE, TIME_IDX)
        USE DEFINE_INPUT_TYPES
        USE DEFINE_STATE_TYPES
        IMPLICIT NONE
        
        CLASS(FL_DS_TYPE),       INTENT(IN)    :: FL_DS
        CLASS(PL_DS_TYPE),       INTENT(IN)    :: PL_DS
        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS

        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(BOXM_STATE_TYPE),  INTENT(INOUT) :: BOXM_STATE
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE

        INTEGER, INTENT(IN) :: TIME_IDX
        
        IF (MOD((TIME_IDX-1)*BOXM_DS%TS_SIM, BOXM_DS%TS_PL) == 0) THEN
            CALL PL_STATE%ADVANCE_GEOM(PL_DS, BOXM_DS, TIME_IDX)
            CALL PL_STATE%BUILD_ACTIVE(BOXM_DS, BOXM_STATE)
            CALL PL_STATE%EMI_TO_PLUMES(FL_DS, PL_DS, BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
            CALL PL_STATE%PROJECT_TO_GRID(PL_DS, BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
            CALL PATCH_STATE%BUILD_ROWS_FROM_W(PL_STATE, BOXM_STATE)
        END IF

        CALL PATCH_STATE%ACCUM_DELTAS_FROM_W(PL_STATE, BOXM_DS, BOXM_STATE)

    END SUBROUTINE PROJECT_PLUMES_TO_GRID

    SUBROUTINE ADVANCE_MET(BOXM_DS, BOXM_STATE, TIME_IDX)
        USE DEFINE_INPUT_TYPES
        USE DEFINE_STATE_TYPES
        IMPLICIT NONE

        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS
        CLASS(BOXM_STATE_TYPE),  INTENT(INOUT) :: BOXM_STATE

        INTEGER, INTENT(IN) :: TIME_IDX

        CALL BOXM_STATE%ADVANCE_MET(BOXM_DS, TIME_IDX)

    END SUBROUTINE ADVANCE_MET

    SUBROUTINE RUN_CHEM(BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
        USE DEFINE_INPUT_TYPES
        USE DEFINE_STATE_TYPES
        USE RUN_CHEM_UTILS
        IMPLICIT NONE

        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS

        CLASS(BOXM_STATE_TYPE),  INTENT(INOUT) :: BOXM_STATE
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE

        INTEGER, INTENT(IN) :: TIME_IDX

        ! Local variables for RUN_LEGACY_CHEM
        INTEGER :: NCELL, NSBOXM, NPP, NPC, NTC, NFL

        ! === LEGACY BACKGROUND CHEMISTRY ===
        CALL BOXM_STATE%RUN_COARSE_BG_CHEM(BOXM_DS)
        
        IF (BOXM_DS%N_AC > 0) THEN
            ! === FINE PLUME CHEMISTRY ===
            CALL PATCH_STATE%RUN_FINE_DELTA_CHEM(BOXM_DS, BOXM_STATE)
        
            ! === COARSE PLUME CHEMISTRY ===
            CALL BOXM_STATE%RUN_COARSE_DELTA_CHEM(BOXM_DS, PATCH_STATE)
        END IF
        
    END SUBROUTINE RUN_CHEM

    SUBROUTINE BACKPROJECT_GRID_TO_PLUMES(FL_DS, PL_DS, BOXM_DS, PL_STATE, BOXM_STATE, PATCH_STATE, TIME_IDX)
        USE DEFINE_INPUT_TYPES
        USE DEFINE_STATE_TYPES
        IMPLICIT NONE

        CLASS(FL_DS_TYPE),       INTENT(IN)    :: FL_DS
        CLASS(PL_DS_TYPE),       INTENT(IN)    :: PL_DS
        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS

        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(BOXM_STATE_TYPE),  INTENT(INOUT) :: BOXM_STATE
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE

        INTEGER, INTENT(IN) :: TIME_IDX

        IF (MOD((TIME_IDX-1)*BOXM_DS%TS_SIM, BOXM_DS%TS_PL) == 0) THEN
            CALL PL_STATE%BACKPROJECT_FROM_GRID(PL_DS, BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
        END IF

    END SUBROUTINE BACKPROJECT_GRID_TO_PLUMES

    SUBROUTINE WRITE_OUTPUTS(PL_OUT, BOXM_OUT, PATCH_TABLE, PL_DS, BOXM_DS, PL_STATE, BOXM_STATE, PATCH_STATE, TIME_IDX)
        USE DEFINE_INPUT_TYPES
        USE DEFINE_OUTPUT_TYPES
        USE DEFINE_STATE_TYPES
        IMPLICIT NONE

        CLASS(PL_DS_TYPE),       INTENT(IN)    :: PL_DS
        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS

        CLASS(PL_OUT_TYPE),      INTENT(INOUT) :: PL_OUT
        CLASS(BOXM_OUT_TYPE),    INTENT(INOUT) :: BOXM_OUT
        CLASS(PATCH_TABLE_TYPE), INTENT(INOUT) :: PATCH_TABLE

        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(BOXM_STATE_TYPE),  INTENT(INOUT) :: BOXM_STATE
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE

        INTEGER, INTENT(IN) :: TIME_IDX
        
        CALL BOXM_OUT%WRITE(BOXM_STATE, TIME_IDX)
        IF (BOXM_DS%N_AC > 0) THEN
            CALL PL_OUT%WRITE(PL_STATE, PL_DS, TIME_IDX)
            CALL PATCH_TABLE%WRITE(PATCH_STATE, BOXM_DS, BOXM_STATE, TIME_IDX)
        END IF

    END SUBROUTINE WRITE_OUTPUTS

    SUBROUTINE RESET_STATES(PL_STATE, BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
        USE HELPERS
        USE DEFINE_INPUT_TYPES
        USE DEFINE_STATE_TYPES
        IMPLICIT NONE

        CLASS(PL_STATE_TYPE),    INTENT(INOUT) :: PL_STATE
        CLASS(BOXM_DS_TYPE),     INTENT(IN)    :: BOXM_DS
        CLASS(BOXM_STATE_TYPE),  INTENT(INOUT) :: BOXM_STATE
        CLASS(PATCH_STATE_TYPE), INTENT(INOUT) :: PATCH_STATE

        INTEGER, INTENT(IN) :: TIME_IDX

        IF (MOD((TIME_IDX-1)*BOXM_DS%TS_SIM, BOXM_DS%TS_PL) == 0) THEN
            IF (ALLOCATED(PL_STATE%AGE_S)) PL_STATE%AGE_S(:) = 0
            IF (ALLOCATED(PL_STATE%LONGITUDE)) PL_STATE%LONGITUDE(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%LONGITUDE_M)) PL_STATE%LONGITUDE_M(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%LATITUDE)) PL_STATE%LATITUDE(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%LATITUDE_M)) PL_STATE%LATITUDE_M(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%ALTITUDE)) PL_STATE%ALTITUDE(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%LEVEL)) PL_STATE%LEVEL(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%WIDTH)) PL_STATE%WIDTH(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%DEPTH)) PL_STATE%DEPTH(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%HEADING)) PL_STATE%HEADING(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%SIGMA_YY)) PL_STATE%SIGMA_YY(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%SIGMA_YZ)) PL_STATE%SIGMA_YZ(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%SIGMA_ZZ)) PL_STATE%SIGMA_ZZ(:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%ACTIVE_SEG_FLAG)) PL_STATE%ACTIVE_SEG_FLAG(:) = .FALSE.
            ! Keep carried plume mass between timesteps; it is updated by BACKPROJECT_GRID_TO_PLUMES
            ! and only newly emitted segments are reset in PL_STATE_ADVANCE_MASS.

            IF (ALLOCATED(PL_STATE%Y_HALF)) PL_STATE%Y_HALF(:,:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%Z_HALF)) PL_STATE%Z_HALF(:,:) = 0.0_DP
            IF (ALLOCATED(PL_STATE%SLICE_POLYS_M)) PL_STATE%SLICE_POLYS_M(:,:,:,:) = 0.0_DP

            
        END IF

        IF (ALLOCATED(BOXM_STATE%TEMP)) BOXM_STATE%TEMP(:) = 0.0_DP
        IF (ALLOCATED(BOXM_STATE%H2O)) BOXM_STATE%H2O(:) = 0.0_DP
        IF (ALLOCATED(BOXM_STATE%M)) BOXM_STATE%M(:) = 0.0_DP
        IF (ALLOCATED(BOXM_STATE%O2)) BOXM_STATE%O2(:) = 0.0_DP
        IF (ALLOCATED(BOXM_STATE%N2)) BOXM_STATE%N2(:) = 0.0_DP
        IF (ALLOCATED(BOXM_STATE%SZA)) BOXM_STATE%SZA(:) = 0.0_DP

        IF (ALLOCATED(BOXM_STATE%ACTIVE_FLAG)) BOXM_STATE%ACTIVE_FLAG(:) = .FALSE.

        IF (ALLOCATED(PATCH_STATE%TIME_REL_S)) PATCH_STATE%TIME_REL_S(:) = 0
        IF (ALLOCATED(PATCH_STATE%ROW_CELL_C)) PATCH_STATE%ROW_CELL_C(:) = 0
        IF (ALLOCATED(PATCH_STATE%ROW_CELL_F)) PATCH_STATE%ROW_CELL_F(:) = 0

    END SUBROUTINE RESET_STATES

    SUBROUTINE CLOSE_DATASETS(FL_DS, PL_DS, BOXM_DS, PL_OUT, BOXM_OUT, PATCH_TABLE)
        USE DEFINE_INPUT_TYPES
        USE DEFINE_OUTPUT_TYPES
        IMPLICIT NONE

        CLASS(FL_DS_TYPE),       INTENT(INOUT) :: FL_DS
        CLASS(PL_DS_TYPE),       INTENT(INOUT) :: PL_DS
        CLASS(BOXM_DS_TYPE),     INTENT(INOUT) :: BOXM_DS

        CLASS(PL_OUT_TYPE),      INTENT(INOUT) :: PL_OUT
        CLASS(BOXM_OUT_TYPE),    INTENT(INOUT) :: BOXM_OUT
        CLASS(PATCH_TABLE_TYPE), INTENT(INOUT) :: PATCH_TABLE

        CALL BOXM_DS%CLOSE()
        IF (BOXM_DS%N_AC > 0) THEN
            CALL FL_DS%CLOSE()
            CALL PL_DS%CLOSE()
        END IF
        
        CALL BOXM_OUT%CLOSE()
        IF (BOXM_DS%N_AC > 0) THEN
            CALL PL_OUT%CLOSE()
            CALL PATCH_TABLE%CLOSE()
        END IF

    END SUBROUTINE CLOSE_DATASETS

END MODULE BOXM_RUN_UTILS

MODULE VALIDATION_UTILS
    USE DEFINE_INPUT_TYPES
    USE DEFINE_STATE_TYPES
    USE HELPERS
    IMPLICIT NONE

    PRIVATE
    PUBLIC :: PRINT_TOTAL_PLUME_MASS, PRINT_TOTAL_PATCH_MASS, CHECK_SEGMENT_MASS_RECOVERY

CONTAINS
    SUBROUTINE CHECK_SEGMENT_MASS_RECOVERY(PL_STATE, PATCH_STATE, BOXM_STATE, BOXM_DS, SEG_ID)
        IMPLICIT NONE

        CLASS(PL_STATE_TYPE),    INTENT(IN) :: PL_STATE
        CLASS(PATCH_STATE_TYPE), INTENT(IN) :: PATCH_STATE
        CLASS(BOXM_STATE_TYPE),  INTENT(IN) :: BOXM_STATE
        CLASS(BOXM_DS_TYPE),     INTENT(IN) :: BOXM_DS
        INTEGER,                 INTENT(IN) :: SEG_ID

        INTEGER :: PL_ID, B, K, K2, P, CELL_C, CELL_F, ROW_IDX
        INTEGER :: NX_F, NY_F, SLICE_ID
        REAL(DP) :: DX_C_M, DY_C_M, DX_F, DY_F, VOL_F
        REAL(DP) :: ORIG_MASS, RECOVERED_MASS, CELL_MASS
        REAL(DP) :: SUM_W_SEG, SUM_W_SLICE, SUM_W_REV
        INTEGER :: MAP_COUNT, MISS_COUNT

        IF (SEG_ID < 1 .OR. SEG_ID > PL_STATE%NSEG) THEN
            PRINT *, 'CHECK_SEGMENT_MASS_RECOVERY: invalid SEG_ID=', SEG_ID
            RETURN
        END IF

        NX_F = MAX(1, NINT(BOXM_DS%HRES_SIM_C / BOXM_DS%HRES_SIM_F))
        NY_F = NX_F

        PRINT *, '================================================'
        PRINT *, 'SEGMENT RECOVERY CHECK: SEG_ID=', SEG_ID

        DO PL_ID = 1, MIN(PL_STATE%NSPL, 3)
            B = PL_STATE%SPECIES_PL_NUM(PL_ID)
            IF (B < 1 .OR. B > PATCH_STATE%NSBOXM) CYCLE
            IF (B > SIZE(BOXM_STATE%MOL_MASS_C)) CYCLE
            IF (BOXM_STATE%MOL_MASS_C(B) <= 0.0_DP) CYCLE

            ORIG_MASS = PL_STATE%PL_MASS(SEG_ID, PL_ID)
            RECOVERED_MASS = 0.0_DP
            SUM_W_SEG = 0.0_DP
            MAP_COUNT = 0
            MISS_COUNT = 0

            PRINT *, '--- PL_ID=', PL_ID, ' BOXM_ID=', B, ' ORIG_MASS=', ORIG_MASS

            DO SLICE_ID = 1, PL_STATE%NSLICES
                SUM_W_SLICE = 0.0_DP
                DO K = 1, PL_STATE%NNZ_MAP
                    IF (PL_STATE%MAP_SEG(K) == SEG_ID .AND. PL_STATE%MAP_SLICE(K) == SLICE_ID) THEN
                        SUM_W_SLICE = SUM_W_SLICE + PL_STATE%MAP_W(K)
                    END IF
                END DO
            END DO

            DO K = 1, PL_STATE%NNZ_MAP
                IF (PL_STATE%MAP_SEG(K) /= SEG_ID) CYCLE

                MAP_COUNT = MAP_COUNT + 1
                CELL_C = PL_STATE%MAP_CELL_C(K)
                CELL_F = PL_STATE%MAP_CELL_F(K)

                ROW_IDX = -1
                DO P = 1, PATCH_STATE%NROWS
                    IF (PATCH_STATE%ROW_CELL_C(P) == CELL_C .AND. PATCH_STATE%ROW_CELL_F(P) == CELL_F) THEN
                        ROW_IDX = P
                        EXIT
                    END IF
                END DO
                IF (ROW_IDX < 1) THEN
                    MISS_COUNT = MISS_COUNT + 1
                    CYCLE
                END IF

                DX_C_M = BOXM_STATE%DX_C_M(CELL_C)
                DY_C_M = BOXM_STATE%DY_C_M(CELL_C)
                DX_F = DX_C_M / REAL(NX_F, DP)
                DY_F = DY_C_M / REAL(NY_F, DP)
                VOL_F = DX_F * DY_F * BOXM_DS%VRES_SIM_F
                IF (VOL_F <= 0.0_DP) CYCLE

                CELL_MASS = PATCH_STATE%Y_DEL_F(ROW_IDX, B) * BOXM_STATE%MOL_MASS_C(B) * &
                            VOL_F * 1.0E6_DP / 6.02214076E23_DP

                SUM_W_SEG = SUM_W_SEG + PL_STATE%MAP_W(K)

                SUM_W_REV = 0.0_DP
                DO K2 = 1, PL_STATE%NNZ_MAP
                    IF (PL_STATE%MAP_CELL_C(K2) == CELL_C .AND. PL_STATE%MAP_CELL_F(K2) == CELL_F) THEN
                        SUM_W_REV = SUM_W_REV + PL_STATE%MAP_W(K2)
                    END IF
                END DO
                IF (SUM_W_REV <= 0.0_DP) CYCLE

                RECOVERED_MASS = RECOVERED_MASS + (PL_STATE%MAP_W(K) / SUM_W_REV) * CELL_MASS
            END DO

            PRINT *, '   MAP_COUNT      =', MAP_COUNT
            PRINT *, '   MISS_COUNT     =', MISS_COUNT
            PRINT *, '   SUM_W_SEG      =', SUM_W_SEG
            PRINT *, '   RECOVERED_MASS =', RECOVERED_MASS
            IF (ABS(ORIG_MASS) > 0.0_DP) THEN
                PRINT *, '   REC/ORIG       =', RECOVERED_MASS / ORIG_MASS
            END IF
        END DO

        PRINT *, '================================================'
    END SUBROUTINE CHECK_SEGMENT_MASS_RECOVERY

    SUBROUTINE PRINT_TOTAL_PLUME_MASS(PL_STATE)
        CLASS(PL_STATE_TYPE), INTENT(IN) :: PL_STATE
        REAL(DP) :: TOTAL_MASS

        IF (.NOT. ALLOCATED(PL_STATE%PL_MASS)) RETURN
        TOTAL_MASS = SUM(PL_STATE%PL_MASS)
        PRINT *, 'Total plume mass:', TOTAL_MASS
    END SUBROUTINE PRINT_TOTAL_PLUME_MASS

    SUBROUTINE PRINT_TOTAL_PATCH_MASS(PATCH_STATE, BOXM_STATE)
        CLASS(PATCH_STATE_TYPE), INTENT(IN) :: PATCH_STATE
        CLASS(BOXM_STATE_TYPE), INTENT(IN) :: BOXM_STATE
        REAL(DP) :: TOTAL_PATCH_MASS
        INTEGER :: P, B
        TOTAL_PATCH_MASS = 0.0_DP
        IF (.NOT. ALLOCATED(PATCH_STATE%Y_DEL_F)) RETURN
        IF (.NOT. ALLOCATED(BOXM_STATE%MOL_MASS_C)) RETURN

        DO P = 1, SIZE(PATCH_STATE%Y_DEL_F, 1)
            DO B = 1, SIZE(PATCH_STATE%Y_DEL_F, 2)
                IF (ISNAN(PATCH_STATE%Y_DEL_F(P, B)) .OR. ISNAN(BOXM_STATE%MOL_MASS_C(B))) THEN
                    PRINT *, 'NaN detected at row', P, 'species', B, &
                        ', Y_DEL_F(P,B)=', PATCH_STATE%Y_DEL_F(P,B), &
                        ', MOL_MASS_C(B)=', BOXM_STATE%MOL_MASS_C(B)
                END IF
            END DO
        END DO
        PRINT *, 'Y_DEL_F shape:', SIZE(PATCH_STATE%Y_DEL_F,1), SIZE(PATCH_STATE%Y_DEL_F,2)
        PRINT *, 'MOL_MASS_C shape:', SIZE(BOXM_STATE%MOL_MASS_C)

        DO P = 1, SIZE(PATCH_STATE%Y_DEL_F, 1)
            DO B = 1, SIZE(PATCH_STATE%Y_DEL_F, 2)
                TOTAL_PATCH_MASS = TOTAL_PATCH_MASS + PATCH_STATE%Y_DEL_F(P, B) * BOXM_STATE%MOL_MASS_C(B)
            END DO
        END DO
        PRINT *, 'Total patch (grid) mass:', TOTAL_PATCH_MASS
    END SUBROUTINE PRINT_TOTAL_PATCH_MASS

END MODULE VALIDATION_UTILS

PROGRAM BOXM_RUN
    USE BOXM_RUN_UTILS
    USE DEFINE_INPUT_TYPES
    USE DEFINE_STATE_TYPES
    USE DEFINE_OUTPUT_TYPES
    USE VALIDATION_UTILS
    IMPLICIT NONE

    TYPE(FL_DS_TYPE)       :: FL_DS
    TYPE(PL_DS_TYPE)       :: PL_DS
    TYPE(BOXM_DS_TYPE)     :: BOXM_DS

    TYPE(PL_STATE_TYPE)    :: PL_STATE
    TYPE(BOXM_STATE_TYPE)  :: BOXM_STATE
    TYPE(PATCH_STATE_TYPE) :: PATCH_STATE

    TYPE(PL_OUT_TYPE)      :: PL_OUT
    TYPE(BOXM_OUT_TYPE)    :: BOXM_OUT
    TYPE(PATCH_TABLE_TYPE) :: PATCH_TABLE

    CHARACTER(LEN=256)     :: JOB_ID
    CHARACTER(LEN=1024)    :: DATA_PATH

    INTEGER :: TIME_IDX, S
    REAL :: PROG_PCT

    ! RETRIEVE JOB ID FROM COMMAND LINE ARG
    CALL GETARG(1, JOB_ID)
    CALL GETARG(2, DATA_PATH)

    CALL BOXM_RUN_INIT(FL_DS, PL_DS, BOXM_DS, PL_STATE, BOXM_STATE, PATCH_STATE, &
                       PL_OUT, BOXM_OUT, PATCH_TABLE, JOB_ID, DATA_PATH)

    ! MAIN SIMULATION LOOP OVER BOXM TIME STEPS
    DO TIME_IDX = 1, BOXM_DS%NTBOXM

        PROG_PCT = 100.0 * REAL(TIME_IDX) / REAL(MAX(1, BOXM_DS%NTBOXM))
        WRITE(*,'(A,I0,A,I0,A,F6.2,A)') "SIM progress: TIME_IDX ", TIME_IDX, " / ", BOXM_DS%NTBOXM, " (", PROG_PCT, "%)"

        CALL RESET_STATES(PL_STATE, BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)

        CALL ADVANCE_MET(BOXM_DS, BOXM_STATE, TIME_IDX)

        IF (BOXM_DS%N_AC > 0) THEN
            
            ! PROJECT_PLUMES_TO_GRID
            IF (MOD((TIME_IDX-1)*BOXM_DS%TS_SIM, BOXM_DS%TS_OUT) == 0) THEN
                CALL PROJECT_PLUMES_TO_GRID(FL_DS, PL_DS, BOXM_DS, PL_STATE, BOXM_STATE, PATCH_STATE, TIME_IDX)
            END IF

            IF (BOXM_DS%RUN_CHEM == 1) THEN
                ! UPDATE_CHEMISTRY
                CALL RUN_CHEM(BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
            END IF

            ! BACKPROJECT_GRID_TO_PLUMES
            IF (MOD((TIME_IDX-1)*BOXM_DS%TS_SIM, BOXM_DS%TS_OUT) == 0) THEN
                CALL BACKPROJECT_GRID_TO_PLUMES(FL_DS, PL_DS, BOXM_DS, PL_STATE, BOXM_STATE, PATCH_STATE, TIME_IDX)
                ! CALL CHECK_SEGMENT_MASS_RECOVERY(PL_STATE, PATCH_STATE, BOXM_STATE, BOXM_DS, 1)
            END IF

        ELSE
            IF (BOXM_DS%RUN_CHEM == 1) THEN
                ! UPDATE_CHEMISTRY
                CALL RUN_CHEM(BOXM_DS, BOXM_STATE, PATCH_STATE, TIME_IDX)
            END IF
        END IF

        IF (MOD((TIME_IDX-1)*BOXM_DS%TS_SIM, BOXM_DS%TS_OUT) == 0) THEN
            ! WRITE BACK TO OUTPUT DATASETS
            CALL WRITE_OUTPUTS(PL_OUT, BOXM_OUT, PATCH_TABLE, PL_DS, BOXM_DS, PL_STATE, BOXM_STATE, PATCH_STATE, TIME_IDX)
        END IF
    END DO
    

    CALL CLOSE_DATASETS(FL_DS, PL_DS, BOXM_DS, PL_OUT, BOXM_OUT, PATCH_TABLE)
    
END PROGRAM BOXM_RUN
