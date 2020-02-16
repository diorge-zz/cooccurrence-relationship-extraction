import pyspark.sql.functions as f
from pyspark.conf import SparkConf
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.sql import SparkSession


spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()


def build_matrix(svo_path: str,
                 cat1_instances: set,
                 cat2_instances: set
                 ) -> CoordinateMatrix:
    raw_df = spark.read.csv(svo_path, sep='\t')

    pairs_df = (raw_df.filter((f.col('_c0').isin(cat1_instances)
                              & f.col('_c2').isin(cat2_instances))
                              | (f.col('_c0').isin(cat2_instances)
                                 & f.col('_c2').isin(cat1_instances)))
                      .rdd
                      .map(lambda x: (tuple(sorted((x['_c0'], x['_c2']))),
                                      x['_c1'],
                                      int(x['_c3'])))
                      .toDF(['pair', 'verb', 'n']))

    named_coords = (pairs_df.selectExpr('pair', 'verb as left_verb', 'n')
                            .join(pairs_df.selectExpr('pair',
                                                      'verb as right_verb'),
                                  'pair')
                            .filter('left_verb < right_verb')
                            .groupby(['left_verb', 'right_verb'])
                            .count())

    verb_to_id = (pairs_df.select('verb')
                          .distinct()
                          .rdd
                          .zipWithIndex()
                          .map(lambda r: [r[0].verb, r[1]])
                          .toDF(['verb', 'id']))

    coords = (named_coords.join(verb_to_id,
                                named_coords.left_verb == verb_to_id.verb)
                          .selectExpr('right_verb',
                                      'id as left_verb_id',
                                      'count')
                          .join(verb_to_id,
                                named_coords.right_verb == verb_to_id.verb)
                          .selectExpr('left_verb_id',
                                      'id as right_verb_id',
                                      'count'))

    matrix = CoordinateMatrix(coords.rdd.map(lambda c: MatrixEntry(*c)))

    return matrix
