from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

from .utils import write_scalar_local, write_list_local, write_df_local

class USVehicleAccidentAnalysis:
    def __init__(self, spark: SparkSession, config: dict):
        self.spark = spark
        self.config = config
        inputs = config.get("INPUT_FILENAME", {})

        # Read all inputs as strings to avoid schema surprises; cast as needed later
        self.charges_df = self._read_csv(inputs.get("Charges"))
        self.damages_df = self._read_csv(inputs.get("Damages"))
        self.endorse_df = self._read_csv(inputs.get("Endorse"))
        self.primary_df = self._read_csv(inputs.get("Primary_Person"))
        self.restrict_df = self._read_csv(inputs.get("Restrict"))
        self.units_df = self._read_csv(inputs.get("Units"))

    def _read_csv(self, path: Optional[str]) -> DataFrame:
        if not path:
            return self.spark.createDataFrame([], schema=None)
        return (
            self.spark.read.option("header", True)
            .option("inferSchema", False)
            .csv(path)
        )

    def _write_scalar(self, value, out_path: Optional[str], fmt: str):
        if out_path and fmt:
            try:
                df = self.spark.createDataFrame([(value,)], ["value"])
                df.coalesce(1).write.mode("overwrite").format(fmt).save(out_path)
            except Exception:
                # Windows without winutils fallback
                write_scalar_local(value, out_path, fmt)
        return value

    def _write_list(self, values: List[str], out_path: Optional[str], fmt: str, col_name: str = "value"):
        if out_path and fmt:
            try:
                df = self.spark.createDataFrame([(v,) for v in values], [col_name])
                df.coalesce(1).write.mode("overwrite").format(fmt).save(out_path)
            except Exception:
                write_list_local(values, out_path, fmt, col_name=col_name)
        return values

    # 1. Find the number of crashes (accidents) in which number of males killed are > 0
    def count_male_accidents(self, out_path: Optional[str], fmt: str):
        df = self.primary_df
        res = (
            df.filter((F.upper(F.col("PRSN_GNDR_ID")) == F.lit("MALE")) & (F.col("DEATH_CNT").cast("int") > 0))
            .select("CRASH_ID")
            .distinct()
            .count()
        )
        return self._write_scalar(res, out_path, fmt)

    # 2. How many two-wheelers are booked for crashes?
    def count_2_wheeler_accidents(self, out_path: Optional[str], fmt: str):
        df = self.units_df
        two_wheelers = df.filter(
            (F.upper(F.col("VEH_BODY_STYL_ID")).like("%MOTORCYCLE%"))
            | (F.upper(F.col("UNIT_DESC_ID")).like("%MOTORCYCLE%"))
            | (F.upper(F.col("VEH_BODY_STYL_ID")).like("%SCOOTER%"))
            | (F.upper(F.col("VEH_BODY_STYL_ID")).like("%MOPED%"))
        )
        res = two_wheelers.count()
        return self._write_scalar(res, out_path, fmt)

    # 3. Top 5 vehicle makes for fatal crashes where airbags did not deploy for the driver
    def top_5_vehicle_makes_for_fatal_crashes_without_airbags(self, out_path: Optional[str], fmt: str):
        p = self.primary_df.alias("p")
        u = self.units_df.alias("u")
        joined = (
            p.join(u, on=["CRASH_ID", "UNIT_NBR"], how="inner")
            .filter((F.col("p.DEATH_CNT").cast("int") > 0))
            .filter(F.upper(F.col("p.PRSN_AIRBAG_ID")).isin("NOT DEPLOYED", "NONE", "NA"))
            .filter(F.upper(F.col("p.PRSN_TYPE_ID")).like("%DRIVER%"))
        )
        res_df = (
            joined.groupBy("VEH_MAKE_ID").count().orderBy(F.col("count").desc(), F.col("VEH_MAKE_ID").asc()).limit(5)
        )
        res = [r[0] for r in res_df.collect()]
        return self._write_list(res, out_path, fmt, col_name="VEH_MAKE_ID")

    # 4. Number of vehicles with valid licenses involved in hit-and-run
    def count_hit_and_run_with_valid_licenses(self, out_path: Optional[str], fmt: str):
        p = self.primary_df
        u = self.units_df
        valid_license = ~F.upper(F.col("DRVR_LIC_TYPE_ID")).isin("UNKNOWN", "NA", "UNLICENSED")
        hit_and_run = F.upper(F.col("VEH_HNR_FL")).isin("Y", "YES", "TRUE", "T")
        joined = p.join(u, on=["CRASH_ID", "UNIT_NBR"], how="inner").filter(valid_license & hit_and_run)
        res = joined.count()
        return self._write_scalar(res, out_path, fmt)

    # 5. State with highest number of accidents with no females involved
    def get_state_with_no_female_accident(self, out_path: Optional[str], fmt: str):
        p = self.primary_df
        # Crashes that have any female
        female_crashes = (
            p.filter(F.upper(F.col("PRSN_GNDR_ID")) == "FEMALE").select("CRASH_ID").distinct()
        )
        # Exclude crashes with any female, then count by driver license state
        no_female = (
            p.join(female_crashes, on="CRASH_ID", how="left_anti")
            .filter(F.col("DRVR_LIC_STATE_ID").isNotNull())
            .groupBy("DRVR_LIC_STATE_ID")
            .agg(F.countDistinct("CRASH_ID").alias("crashes"))
            .orderBy(F.col("crashes").desc())
        )
        top = no_female.limit(1).collect()
        res = top[0][0] if top else None
        return self._write_scalar(res, out_path, fmt)

    # 6. Top 3rd to 5th vehicle makes contributing to most injuries including deaths
    def get_top_vehicle_contributing_to_injuries(self, out_path: Optional[str], fmt: str):
        u = self.units_df
        injuries = (
            u.withColumn("TOT_INJRY_CNT", F.col("TOT_INJRY_CNT").cast("int"))
             .withColumn("DEATH_CNT", F.col("DEATH_CNT").cast("int"))
             .withColumn("injuries", (F.coalesce(F.col("TOT_INJRY_CNT"), F.lit(0)) + F.coalesce(F.col("DEATH_CNT"), F.lit(0))))
        )
        agg = injuries.groupBy("VEH_MAKE_ID").agg(F.sum("injuries").alias("total_injuries"))
        w = Window.orderBy(F.col("total_injuries").desc(), F.col("VEH_MAKE_ID").asc())
        ranked = agg.withColumn("rn", F.row_number().over(w)).filter((F.col("rn") >= 3) & (F.col("rn") <= 5)).orderBy("rn")
        res = [r[0] for r in ranked.collect()]
        return self._write_list(res, out_path, fmt, col_name="VEH_MAKE_ID")

    # 7. For all body styles, the top ethnic user group for each unique body style
    def get_top_ethnic_ug_crash_for_each_body_style(self, out_path: Optional[str], fmt: str) -> DataFrame:
        p = self.primary_df
        u = self.units_df
        joined = p.join(u, on=["CRASH_ID", "UNIT_NBR"], how="inner")
        grp = joined.groupBy("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID").agg(F.count(F.lit(1)).alias("cnt"))
        w = Window.partitionBy("VEH_BODY_STYL_ID").orderBy(F.col("cnt").desc(), F.col("PRSN_ETHNICITY_ID").asc())
        top_per_style = grp.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")
        # Write if requested
        if out_path and fmt:
            try:
                top_per_style.coalesce(1).write.mode("overwrite").format(fmt).save(out_path)
            except Exception:
                write_df_local(top_per_style, out_path, fmt)
        return top_per_style

    # 8. Top 5 driver zip codes with alcohol as contributing factor among crashed cars
    def get_top_5_zip_codes_with_alcohols_as_cf_for_crash(self, out_path: Optional[str], fmt: str):
        p = self.primary_df
        u = self.units_df
        # Consider car-like body styles
        car_like = (
            F.upper(F.col("VEH_BODY_STYL_ID")).like("%CAR%") | F.upper(F.col("VEH_BODY_STYL_ID")).like("%SPORT UTILITY%") | F.upper(F.col("VEH_BODY_STYL_ID")).like("%STATION WAGON%")
        )
        alcohol = F.upper(F.col("PRSN_ALC_RSLT_ID")).isin("POSITIVE", "YES", "Y")
        joined = p.join(u, on=["CRASH_ID", "UNIT_NBR"], how="inner").filter(car_like & alcohol & F.col("DRVR_ZIP").isNotNull())
        agg = joined.groupBy("DRVR_ZIP").agg(F.countDistinct("CRASH_ID").alias("crashes")).orderBy(F.col("crashes").desc())
        res = [r[0] for r in agg.limit(5).collect()]
        return self._write_list(res, out_path, fmt, col_name="DRVR_ZIP")

    # 9. Count of distinct Crash IDs where No Damaged Property observed and Damage Level > 4 and car avails insurance
    def get_crash_ids_with_no_damage(self, out_path: Optional[str], fmt: str):
        d = self.damages_df
        u = self.units_df
        no_damage_property = d.filter(F.upper(F.col("DAMAGED_PROPERTY")).isin("NONE", "NO DAMAGE", "N/A", "NA"))
        # Extract numeric scale from e.g. 'DAMAGED 5' or '5' and compare > 4; tolerate malformed values
        def scale(colname):
            digits = F.regexp_extract(F.upper(F.col(colname)), r"(\d+)", 1)
            return F.when(F.length(digits) > 0, digits.cast("int")).otherwise(F.lit(None))
        max_scale = F.greatest(scale("VEH_DMAG_SCL_1_ID"), scale("VEH_DMAG_SCL_2_ID"))
        dmg_scale_ok = (F.coalesce(max_scale, F.lit(0)) > 4)
        insured = ~F.upper(F.col("FIN_RESP_TYPE_ID")).isin("NA", "UNKNOWN", "NONE")
        joined = no_damage_property.select("CRASH_ID").distinct().join(u, on="CRASH_ID", how="inner").filter(dmg_scale_ok & insured)
        res = joined.select("CRASH_ID").distinct().count()
        return self._write_scalar(res, out_path, fmt)

    # 10. Top 5 vehicle makes under speeding offences, licensed drivers, top 10 colors, and top 25 offence states
    def get_top_5_vehicle_brand(self, out_path: Optional[str], fmt: str):
        c = self.charges_df
        p = self.primary_df
        u = self.units_df

        speeding = c.filter(F.upper(F.col("CHARGE")).like("%SPEED%"))
        licensed = p.filter(~F.upper(F.col("DRVR_LIC_TYPE_ID")).isin("NA", "UNKNOWN", "UNLICENSED"))

        # Top 25 offence states (by charges joined to person state)
        charges_with_state = speeding.join(p.select("CRASH_ID", "UNIT_NBR", "PRSN_NBR", "DRVR_LIC_STATE_ID"), on=["CRASH_ID", "UNIT_NBR", "PRSN_NBR"], how="inner")
        top_states = (
            charges_with_state
            .groupBy("DRVR_LIC_STATE_ID")
            .agg(F.count(F.lit(1)).alias("cnt"))
            .orderBy(F.col("cnt").desc())
            .limit(25)
            .select("DRVR_LIC_STATE_ID")
        )

        # Top 10 vehicle colors
        top_colors = (
            u.groupBy("VEH_COLOR_ID").agg(F.count(F.lit(1)).alias("cnt")).orderBy(F.col("cnt").desc()).limit(10).select("VEH_COLOR_ID")
        )

        # Combine constraints
        joined = (
            speeding
            .select("CRASH_ID", "UNIT_NBR")
            .dropDuplicates()
            .join(licensed.select("CRASH_ID", "UNIT_NBR", "DRVR_LIC_STATE_ID"), on=["CRASH_ID", "UNIT_NBR"], how="inner")
            .join(u.select("CRASH_ID", "UNIT_NBR", "VEH_MAKE_ID", "VEH_COLOR_ID"), on=["CRASH_ID", "UNIT_NBR"], how="inner")
            .join(top_states.withColumnRenamed("DRVR_LIC_STATE_ID", "STATE_ID"), on=(F.col("DRVR_LIC_STATE_ID") == F.col("STATE_ID")), how="inner")
            .join(top_colors.withColumnRenamed("VEH_COLOR_ID", "COLOR_ID"), on=(F.col("VEH_COLOR_ID") == F.col("COLOR_ID")), how="inner")
        )

        res_df = joined.groupBy("VEH_MAKE_ID").agg(F.countDistinct("CRASH_ID").alias("cnt")).orderBy(F.col("cnt").desc()).limit(5)
        res = [r[0] for r in res_df.collect()]
        return self._write_list(res, out_path, fmt, col_name="VEH_MAKE_ID")

